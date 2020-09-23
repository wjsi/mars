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
from collections import deque

from ..operands import Fetch, ShuffleProxy
from ..graph import DAG
from .core import MockThreadPoolExecutor, OperandState
from .analyzer import GraphAnalyzer
from .data_tracker import DataTracker

try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = type('UFuncTypeError', (Exception,), {})

try:
    import gevent.event
except ImportError:  # pragma: no cover
    gevent = None

logger = logging.getLogger(__name__)


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


class OperandProfile:
    __slots__ = 'ops', 'state', 'is_target', 'listeners'

    def __init__(self, ops=None, state=None, listeners=None):
        self.ops = ops or set()
        self.state = state or OperandState.UNSCHEDULED
        self.is_target = False
        self.listeners = listeners or set()


class ChunkGraphExecutor:
    def __init__(self, sync_provider, engine=None):
        self._graph = DAG()
        self._runtime_graph = DAG()
        self._target_chunk_keys = set()
        self._active_chunks = set()
        self._engine = engine

        self._sync_provider = sync_provider
        self._lock = sync_provider.rlock()

        self._op_profiles = dict()  # type: dict[str, OperandProfile]

        self._data_tracker = DataTracker()

    def _order_starts(self, starts):
        visited = set()
        op_keys = set()
        starts = deque(starts)
        if not starts:
            return

        stack = deque([starts.popleft()])

        while stack:
            node = stack.popleft()
            if node not in visited:
                preds = self._runtime_graph.predecessors(node)
                if not preds or all(pred in visited for pred in preds):
                    if len(preds) == 0:
                        op_key = node.op.key
                        if op_key not in op_keys:
                            op_keys.add(op_key)
                            yield node.op
                    visited.add(node)
                    stack.extend(n for n in self._runtime_graph[node] if n not in visited)
                else:
                    stack.appendleft(node)
                    stack.extendleft(reversed(list(n for n in self._runtime_graph.predecessors(node)
                                                   if n not in visited)))
            if not stack and starts:
                stack.appendleft(starts.popleft())

    def _notify_op_listeners(self, op_key, exc_info=None):
        op_profile = self._op_profiles[op_key]
        if not op_profile.is_target:
            return
        for listener in op_profile.listeners:
            listener.finish_operand(op_key, exc_info=exc_info)

    def _collect_initial_outputs(self, graph):
        ops = []
        for c in graph.iter_indep():
            if self._op_profiles[c.op.key].state == OperandState.UNSCHEDULED:
                ops.append(c)
        return ops

    def _build_runtime_graph(self):
        runtime_graph = DAG()
        stored_targets = self.filter_stored_data(self._target_chunk_keys)
        self._target_chunk_keys.difference_update(stored_targets)
        stop_keys = set(self._data_tracker.tracked_keys) | set(stored_targets)
        for profile in self._op_profiles.values():
            if profile.state != OperandState.RUNNING:
                continue
            for op in profile.ops:
                stop_keys |= set(c.key for c in op.outputs)

        queue = deque(c for c in self._graph if self._graph.count_successors(c) == 0
                      and c.key in self._target_chunk_keys and c.key not in stop_keys)
        while queue:
            item = queue.popleft()
            runtime_graph.add_node(item)

            for pred in self._graph.iter_predecessors(item):
                if pred.key not in stop_keys:
                    queue.append(pred)

        for n in runtime_graph:
            for succ in self._graph.iter_successors(n):
                if succ in runtime_graph:
                    runtime_graph.add_edge(n, succ)

        self._runtime_graph = runtime_graph

    def submit_chunk_graph(self, graph, chunk_keys=None):
        with self._lock:
            chunk_keys = chunk_keys or set(c.key for c in graph if graph.count_predecessors(c) == 0)
            self._target_chunk_keys.update(chunk_keys)

            if len(self._graph) == 0:
                graph.copyto(self._graph)
            else:
                key_to_chunks = dict((c.key, c) for c in self._graph)
                for n in graph.topological_iter():
                    self._graph.add_node(n)
                    key_to_chunks[n.key] = n
                    for pred in graph.iter_predecessors(n):
                        pred = key_to_chunks[pred.key]
                        self._graph.add_edge(pred, n)
            self._build_runtime_graph()
            target_chunks = [n for n in self._runtime_graph
                             if self._runtime_graph.count_successors(n) == 0]

        for c in self._runtime_graph.dfs(target_chunks, visit_predicate='all', reverse=True):
            try:
                op_profile = self._op_profiles[c.op.key]
                op_profile.ops.add(c.op)
                if op_profile.state in (OperandState.FINISHED, OperandState.FREED):
                    op_profile.state = OperandState.UNSCHEDULED
            except KeyError:
                self._op_profiles[c.op.key] = OperandProfile(
                    {c.op}, OperandState.UNSCHEDULED, set())

        ops_to_run = set(n.op.key for n in target_chunks)
        listener = GraphListener(ops_to_run, self._sync_provider)
        for key in ops_to_run:
            self._op_profiles[key].listeners.add(listener)

        for c in target_chunks:
            self._op_profiles[c.op.key].is_target = True

        keys = []
        start_op_keys = list(
            self._order_starts(self._collect_initial_outputs(self._runtime_graph)))
        self.assign_devices(start_op_keys)

        self._data_tracker.update_graph(self._runtime_graph, self._target_chunk_keys)

        for op in start_op_keys:
            keys.append(op.key)
        self.enqueue_operands(keys[::-1])

        self.submit_operands()
        return listener

    def finish_operand(self, op_key):
        with self._lock:
            if self._op_profiles[op_key].state == OperandState.FATAL:
                return
            self._op_profiles[op_key].state = OperandState.FINISHED

        ops = list(self._op_profiles[op_key].ops)
        # note that currently execution is the chunk-level
        # so we pass the last operand's first output to Executor.handle
        last_op = ops[-1]

        refs = self._data_tracker.add_tracks(last_op.outputs)

        output_keys = set([c.key for c in last_op.outputs])
        # handle other operands
        for rest_op in ops[:-1]:
            for op_output, rest_op_output in zip(last_op.outputs, rest_op.outputs):
                # if the op's outputs have been stored,
                # other same key ops' results will be the same
                if rest_op_output.key not in output_keys:
                    self.copy_data_ref(rest_op_output.key, op_output.key)
                    output_keys.add(rest_op_output.key)

        op_keys_to_execute = []
        chunk_keys_to_delete = set()
        for output in itertools.chain(*[op.outputs for op in ops]):
            # the output not in the graph will be skipped
            with self._lock:
                if output not in self._runtime_graph:
                    continue
                # remove targets
                self._target_chunk_keys.difference_update([output.key])
                # in case that operand has multiple outputs
                # and some of the output not in result keys, delete them
                if refs.get(output.key) == 0:
                    chunk_keys_to_delete.add(output.key)

                # add successors' operands to queue
                successors = self._runtime_graph.successors(output)
                for succ_chunk in successors:
                    preds = self._runtime_graph.predecessors(succ_chunk)
                    succ_op_key = succ_chunk.op.key
                    succ_state = self._op_profiles[succ_op_key].state
                    if succ_state not in (OperandState.READY, OperandState.RUNNING, OperandState.FINISHED) \
                            and (len(preds) == 0
                                 or all(self._op_profiles[pred.op.key].state in OperandState.SUCCESSFUL_STATES
                                        for pred in preds)):
                        op_keys_to_execute.append(succ_op_key)
                        self._op_profiles[succ_op_key].state = OperandState.READY

        # clean the predecessors' results if ref counts equals 0
        with self._lock:
            chunk_keys_to_delete |= self._data_tracker.decref(ops)
            for output in itertools.chain(*[op.outputs for op in ops]):
                for input_chunk in output.op.inputs or ():
                    if all(c.key not in self._data_tracker and c.key not in self._target_chunk_keys
                           for c in input_chunk.op.outputs):
                        self._op_profiles[input_chunk.op.key].state = OperandState.FREED

        self.enqueue_operands(op_keys_to_execute)
        self.submit_operands()
        if not isinstance(last_op, ShuffleProxy):
            self.delete_data_keys(chunk_keys_to_delete)
        self._notify_op_listeners(op_key)

    def set_operand_to_fail(self, op_key, exc_info=None):
        with self._lock:
            self._op_profiles[op_key].state = OperandState.FATAL

            op_chunks = dict()
            for op in self._op_profiles[op_key].ops:
                op_chunks.update({c.key: c for c in op.outputs})

            keys_to_cancel = set()
            for c in self._runtime_graph.dfs(list(op_chunks.values())):
                keys_to_cancel.add(c.op.key)

        for op_key in keys_to_cancel:
            self.cancel_operand(op_key, exc_info=exc_info)
            with self._lock:
                self._op_profiles[op_key].state = OperandState.CANCELLED
            self._notify_op_listeners(op_key, exc_info=exc_info)

    def assign_devices(self, initial_op_keys):
        raise NotImplementedError

    def enqueue_operands(self, op_keys):
        raise NotImplementedError

    def submit_operands(self):
        raise NotImplementedError

    def copy_data_ref(self, dest_chunk_key, src_chunk_key):
        raise NotImplementedError

    def cancel_operand(self, op_key, exc_info=None):
        pass

    def delete_data_keys(self, chunk_keys):
        raise NotImplementedError

    def filter_stored_data(self, chunk_keys):
        raise NotImplementedError


class LocalChunkGraphExecutor(ChunkGraphExecutor):
    _method_name = 'execute'
    _op_runners = dict()

    def __init__(self, chunk_results, sync_provider, engine=None, n_parallel=None):
        super().__init__(sync_provider, engine=engine)
        self._chunk_results = chunk_results

        self._n_parallel = n_parallel or 1
        self._n_slots = self._n_parallel

        self._op_stack = []

        # pool executor for the operand execution
        self._operand_executor = sync_provider.thread_pool_executor(self._n_parallel)

    def assign_devices(self, initial_op_keys):
        if self._n_parallel <= 1 or self._engine != 'cupy':
            return

        devices = list(range(self._n_parallel))
        analyzer = GraphAnalyzer(self._runtime_graph, {k: 1 for k in devices})
        assignments = analyzer.calc_operand_assignments(initial_op_keys)
        for k, v in assignments.items():
            for ops in self._op_profiles[k].ops:
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
            raise KeyError(f'No handler found for op: {op}')

    def _execute_operand(self, op_key):
        try:
            ops = list(self._op_profiles[op_key].ops)

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
        finally:
            with self._lock:
                self._n_slots += 1

    def enqueue_operands(self, op_keys):
        with self._lock:
            self._op_stack.extend(reversed(op_keys))

    def submit_operands(self):
        with self._lock:
            op_keys = self._op_stack[-self._n_slots:]
            del self._op_stack[-self._n_slots:]
            self._n_slots -= len(op_keys)

        def build_callback(k):
            return lambda _: self.finish_operand(k)

        for op_key in op_keys:
            self._op_profiles[op_key].state = OperandState.RUNNING
            future = self._operand_executor.submit(self._execute_operand, op_key)
            if callable(getattr(future, 'add_done_callback', None)):
                future.add_done_callback(build_callback(op_key))
            else:
                future.rawlink(build_callback(op_key))

    def copy_data_ref(self, dest_chunk_key, src_chunk_key):
        self._chunk_results[dest_chunk_key] = self._chunk_results[src_chunk_key]

    def delete_data_keys(self, chunk_keys):
        for chunk_key in chunk_keys:
            del self._chunk_results[chunk_key]

    def filter_stored_data(self, chunk_keys):
        return [k for k in chunk_keys if k in self._chunk_results]


class MockChunkGraphExecutor(LocalChunkGraphExecutor):
    _method_name = 'estimate_size'
    _op_runners = dict()

    def __init__(self, *args, **kwargs):
        self._mock_max_memory = kwargs.pop('mock_max_memory', 0)
        self._no_intermediate = kwargs.pop('no_intermediate', False)

        self._fetch_keys = set()

        super().__init__(*args, **kwargs)
        self._operand_executor = MockThreadPoolExecutor(1)

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
