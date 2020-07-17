# Copyright 1999-2020 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import logging
import sys
from collections import deque, defaultdict
from enum import Enum

from ..operands import Fetch, ShuffleProxy
from ..graph import DAG
from ..utils import classproperty
from .core import MockThreadPoolExecutor
from .analyzer import GraphAnalyzer

try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = type('UFuncTypeError', (Exception,), {})

try:
    import gevent.event
except ImportError:  # pragma: no cover
    gevent = None

logger = logging.getLogger(__name__)


class GraphState(Enum):
    UNSCHEDULED = 'unscheduled'
    PREPARING = 'preparing'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLING = 'cancelling'
    CANCELLED = 'cancelled'

    @classproperty
    def TERMINATED_STATES(self):
        """
        States on which the graph has already terminated
        """
        return self.SUCCEEDED, self.FAILED


class GraphListener:
    def __init__(self, op_keys, sync_provider=None):
        self._op_keys = set(op_keys)
        self._finished_keys = set()
        self._sync_provider = sync_provider
        self._event = sync_provider.event()
        self._exc_info = None

        if not self._op_keys:
            self._event.set()

    def finish_operand(self, op_key, exc_info=None):
        self._finished_keys.add(op_key)
        self._exc_info = self._exc_info or exc_info
        if exc_info is not None or self._finished_keys == self._op_keys:
            self._event.set()

    def wait(self, timeout=None):
        aux_event = None
        if type(self._event).__module__.startswith('gevent'):
            aux_event = gevent.event.Event()

            def _branch_coro():
                delay = 0.0005
                while not aux_event.is_set():
                    aux_event.wait(timeout=delay)
                    delay = min(delay * 2, 0.05)

            gevent.spawn(_branch_coro)

        try:
            if not self._event.wait(timeout):
                raise TimeoutError
        finally:
            if aux_event is not None:
                aux_event.set()

        if self._exc_info is not None:
            raise self._exc_info[1].with_traceback(self._exc_info[2]) from None


class ChunkGraphExecutor:
    def __init__(self, sync_provider, engine=None):
        self._graph = DAG()
        self._target_chunk_keys = set()
        self._engine = engine

        self._chunk_key_ref_counts = dict()
        self._op_key_listeners = defaultdict(list)

        self._sync_provider = sync_provider
        self._lock = sync_provider.rlock()
        self._op_key_to_ops = dict()
        self._submitted_op_keys = set()
        self._executed_op_keys = set()
        self._cancelled_op_keys = set()

        self._exc_info = None

    def _order_starts(self):
        visited = set()
        op_keys = set()
        starts = deque(self._graph.iter_indep())
        if not starts:
            return

        stack = deque([starts.popleft()])

        while stack:
            node = stack.popleft()
            if node not in visited:
                preds = self._graph.predecessors(node)
                if not preds or all(pred in visited for pred in preds):
                    if len(preds) == 0:
                        op_key = node.op.key
                        if op_key not in op_keys:
                            op_keys.add(op_key)
                            yield node.op
                    visited.add(node)
                    stack.extend(n for n in self._graph[node] if n not in visited)
                else:
                    stack.appendleft(node)
                    stack.extendleft(reversed(list(n for n in self._graph.predecessors(node)
                                                   if n not in visited)))
            if not stack and starts:
                stack.appendleft(starts.popleft())

    def _calc_ref_counts(self):
        for chunk in self._graph:
            for dep_key in chunk.op.get_dependent_data_keys():
                if dep_key in self._target_chunk_keys:
                    # only record ref count for those not in results
                    continue
                self._chunk_key_ref_counts[dep_key] = self._chunk_key_ref_counts.get(dep_key, 0) + 1

    def _calc_op_key_to_ops(self):
        op_key_to_ops = defaultdict(set)

        for chunk in self._graph:
            op_key_to_ops[chunk.op.key].add(chunk.op)

        return op_key_to_ops

    def _notify_op_listeners(self, op_key, exc_info=None):
        for listener in self._op_key_listeners.get(op_key, []):
            listener.finish_operand(op_key, exc_info=exc_info)

    def submit_chunk_graph(self, graph, chunk_keys=None):
        if len(self._graph) == 0:
            graph.copyto(self._graph)
        else:  # pragma: no cover
            raise NotImplementedError

        self._op_key_to_ops = self._calc_op_key_to_ops()
        self._target_chunk_keys.update(chunk_keys)
        self._calc_ref_counts()

        ops_to_run = set(n.op.key for n in graph if graph.count_successors(n) == 0)
        listener = GraphListener(ops_to_run, self._sync_provider)
        for k in ops_to_run:
            self._op_key_listeners[k].append(listener)

        for op in self._order_starts():
            with self._lock:
                self._submitted_op_keys.add(op.key)
            self.submit_operand(op.key)

        return listener

    def finish_operand(self, op_key):
        executed_op_keys = self._executed_op_keys
        ref_counts = self._chunk_key_ref_counts
        deleted_chunk_keys = set()

        with self._lock:
            self._submitted_op_keys.remove(op_key)
            executed_op_keys.add(op_key)

        ops = list(self._op_key_to_ops[op_key])
        # note that currently execution is the chunk-level
        # so we pass the first operand's first output to Executor.handle
        first_op = ops[0]

        first_output_keys = set([c.key for c in first_op.outputs])
        # handle other operands
        for rest_op in ops[1:]:
            for op_output, rest_op_output in zip(first_op.outputs, rest_op.outputs):
                # if the op's outputs have been stored,
                # other same key ops' results will be the same
                if rest_op_output.key not in first_output_keys:
                    self.copy_data_ref(rest_op_output.key, op_output.key)

        for output in itertools.chain(*[op.outputs for op in ops]):
            # the output not in the graph will be skipped
            if output not in self._graph:
                continue
            with self._lock:
                # in case that operand has multiple outputs
                # and some of the output not in result keys, delete them
                if ref_counts.get(output.key) == 0:
                    # if the result has been deleted, it should be skipped
                    if output.key not in deleted_chunk_keys:
                        deleted_chunk_keys.add(output.key)
                        self.delete_data(output.key)

            # clean the predecessors' results if ref counts equals 0
            for dep_key in output.op.get_dependent_data_keys():
                with self._lock:
                    if dep_key in ref_counts:
                        ref_counts[dep_key] -= 1
                        if ref_counts[dep_key] == 0:
                            self.delete_data(dep_key)
                            del ref_counts[dep_key]

            # add successors' operands to queue
            for succ_chunk in self._graph.iter_successors(output):
                preds = self._graph.predecessors(succ_chunk)
                with self._lock:
                    succ_op_key = succ_chunk.op.key
                    if succ_op_key not in self._submitted_op_keys \
                            and succ_op_key not in self._executed_op_keys \
                            and (len(preds) == 0 or all(pred.op.key in executed_op_keys for pred in preds)):
                        self._submitted_op_keys.add(succ_op_key)
                        self.submit_operand(succ_op_key)

        self._notify_op_listeners(op_key)

    def set_operand_to_fail(self, op_key, exc_info=None):
        with self._lock:
            op_chunks = dict()
            for op in self._op_key_to_ops[op_key]:
                op_chunks.update({c.key: c for c in op.outputs})

            keys_to_cancel = set()
            for c in self._graph.dfs(list(op_chunks.values())):
                keys_to_cancel.add(c.op.key)

        for op_key in keys_to_cancel:
            self.cancel_operand(op_key, exc_info=exc_info)
            with self._lock:
                self._cancelled_op_keys.add(op_key)
            self._notify_op_listeners(op_key, exc_info=exc_info)

    def submit_operand(self, op_key):
        raise NotImplementedError

    def copy_data_ref(self, dest_chunk_key, src_chunk_key):
        raise NotImplementedError

    def cancel_operand(self, op_key, exc_info=None):
        pass

    def delete_data(self, chunk_key):
        raise NotImplementedError


class LocalChunkGraphExecutor(ChunkGraphExecutor):
    _method_name = 'execute'
    _op_runners = dict()

    def __init__(self, chunk_results, sync_provider, engine=None, n_parallel=None):
        super().__init__(sync_provider, engine=engine)
        self._chunk_results = chunk_results

        self._n_parallel = n_parallel or 1

        # pool executor for the operand execution
        self._operand_executor = sync_provider.thread_pool_executor(self._n_parallel)

    def _assign_devices(self):
        if self._n_parallel <= 1 or self._engine != 'cupy':
            return

        devices = list(range(self._n_parallel))
        analyzer = GraphAnalyzer(self._graph, {k: 1 for k in devices})
        assignments = analyzer.calc_operand_assignments(self._submitted_op_keys)
        for k, v in assignments.items():
            for ops in self._op_key_to_ops[k]:
                for op in ops:
                    op._device = v

    def _call_runner(self, op):
        op_runners = self._op_runners
        try:
            runner = op_runners[type(op)]
        except KeyError:
            runner = getattr(op, self._method_name)

        try:
            return runner(self._chunk_results, op)
        except NotImplementedError:
            for op_cls in op_runners.keys():
                if isinstance(op, op_cls):
                    runner = op_runners[type(op)] = op_runners[op_cls]
                    return runner(self._chunk_results, op)
            raise KeyError('No handler found for op: %s' % op)

    def _execute_operand(self, op_key):
        try:
            ops = list(self._op_key_to_ops[op_key])

            # Cast `UFuncTypeError` to `TypeError` since subclasses of the former is unpickleable.
            # The `UFuncTypeError` was introduced by numpy#12593 since v1.17.0.
            try:
                self._call_runner(ops[0])
            except UFuncTypeError as e:
                raise TypeError(str(e)).with_traceback(sys.exc_info()[2]) from None
        except:  # noqa: E722
            self._exc_info = sys.exc_info()
            self.set_operand_to_fail(op_key, exc_info=self._exc_info)
            raise

    def submit_chunk_graph(self, graph, chunk_keys=None):
        self._assign_devices()
        listener = super().submit_chunk_graph(graph, chunk_keys=chunk_keys)
        return listener

    def submit_operand(self, op_key):
        future = self._operand_executor.submit(self._execute_operand, op_key)
        if callable(getattr(future, 'add_done_callback', None)):
            future.add_done_callback(lambda _: self.finish_operand(op_key))
        else:
            future.rawlink(lambda _: self.finish_operand(op_key))

    def copy_data_ref(self, dest_chunk_key, src_chunk_key):
        self._chunk_results[dest_chunk_key] = self._chunk_results[src_chunk_key]

    def delete_data(self, chunk_key):
        del self._chunk_results[chunk_key]


class MockChunkGraphExecutor(LocalChunkGraphExecutor):
    _method_name = 'estimate_size'
    _op_runners = dict()

    def __init__(self, *args, **kwargs):
        self._mock_max_memory = kwargs.pop('mock_max_memory', 0)
        self._no_intermediate = kwargs.pop('no_intermediate', False)

        self._fetch_keys = set()

        super().__init__(*args, **kwargs)
        self._operand_executor = MockThreadPoolExecutor(self._n_parallel)

    def submit_chunk_graph(self, graph, chunk_keys=None):
        self._fetch_keys.update(v.key for v in graph if isinstance(v.op, Fetch))
        for c in graph:
            if graph.count_predecessors(c) != 0:
                continue
            self._fetch_keys.update(inp.key for inp in c.inputs or ())

        return super().submit_chunk_graph(graph, chunk_keys=chunk_keys)

    def _call_runner(self, op):
        super()._call_runner(op)
        results = self._chunk_results

        output_keys = set(o.key for o in op.outputs or ())

        cur_memory = sum(results[op_output.key][1] for op_output in op.outputs
                         if results.get(op_output.key) is not None)
        if not self._no_intermediate:
            cur_memory += sum(tp[0] for key, tp in results.items()
                              if key not in self._fetch_keys and key not in output_keys
                              and isinstance(tp, tuple))
        self._mock_max_memory = max(cur_memory, self._mock_max_memory)


def ignore(*_):
    pass


LocalChunkGraphExecutor._op_runners[Fetch] = ignore
LocalChunkGraphExecutor._op_runners[ShuffleProxy] = ignore


def register(op_cls, handler=None, size_estimator=None):
    if handler:
        LocalChunkGraphExecutor._op_runners[op_cls] = handler
    if size_estimator:
        MockChunkGraphExecutor._op_runners[op_cls] = size_estimator


def register_default(op_cls):
    LocalChunkGraphExecutor._op_runners.pop(op_cls, None)
    MockChunkGraphExecutor._op_runners.pop(op_cls, None)
