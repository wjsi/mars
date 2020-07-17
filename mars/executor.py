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
import threading
import weakref
import operator
import contextlib
from collections import deque, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from numbers import Integral

import numpy as np

from .operands import Fetch, ShuffleProxy
from .graph import DAG
from .config import options
from .tiles import IterativeChunkGraphBuilder, ChunkGraphBuilder, get_tiled
from .optimizes.runtime.core import RuntimeOptimizer
from .optimizes.tileable_graph import tileable_optimized, OptimizeIntegratedTileableGraphBuilder
from .graph_builder import TileableGraphBuilder
from .context import LocalContext
from .utils import enter_mode, build_fetch, calc_nsplits, has_unknown_shape, prune_chunk_graph

try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = type('UFuncTypeError', (Exception,), {})

try:
    import gevent.event
except ImportError:  # pragma: no cover
    gevent = None

if gevent:
    from .actors.pool.gevent_pool import GeventThreadPool

logger = logging.getLogger(__name__)


class ExecutorSyncProvider(object):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        raise NotImplementedError

    @classmethod
    def semaphore(cls, value):
        raise NotImplementedError

    @classmethod
    def lock(cls):
        raise NotImplementedError

    @classmethod
    def rlock(cls):
        raise NotImplementedError

    @classmethod
    def event(cls):
        raise NotImplementedError

    @classmethod
    def queue(cls, *args, **kwargs):
        raise NotImplementedError


class EventQueue(list):
    def __init__(self, event_cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if event_cls is not None:
            self._has_value = event_cls()
            if len(self) > 0:
                self._has_value.set()
        else:
            self._has_value = None

    def append(self, item):
        super().append(item)
        if self._has_value is not None:
            self._has_value.set()

    def insert(self, index: int, item) -> None:
        super().insert(index, item)
        if self._has_value is not None:
            self._has_value.set()

    def pop(self, index=-1):
        item = super().pop(index)
        if self._has_value is not None and len(self) == 0:
            self._has_value.clear()
        return item

    def clear(self) -> None:
        super().clear()
        if self._has_value is not None:
            self._has_value.clear()

    def wait(self, timeout=None):
        if self._has_value is not None:
            self._has_value.wait(timeout)

    def errored(self):
        if self._has_value is not None:
            self._has_value.set()


class ThreadExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return ThreadPoolExecutor(n_workers)

    @classmethod
    def semaphore(cls, value):
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        return threading.Lock()

    @classmethod
    def rlock(cls):
        return threading.RLock()

    @classmethod
    def event(cls):
        return threading.Event()

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(threading.Event, *args, **kwargs)


class GeventExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return GeventThreadPool(n_workers)

    @classmethod
    def semaphore(cls, value):
        # as gevent threadpool is the **real** thread, so use threading.Semaphore
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        # as gevent threadpool is the **real** thread, so use threading.Lock
        return threading.Lock()

    @classmethod
    def rlock(cls):
        # as gevent threadpool is the **real** thread, so use threading.RLock
        return threading.RLock()

    @classmethod
    def event(cls):
        # as gevent threadpool is the **real** thread, so use threading.Event
        import gevent.event
        return gevent.event.Event()

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(threading.Event, *args, **kwargs)


class MockThreadPoolExecutor(object):
    class _MockResult(object):
        def __init__(self, result=None, exc_info=None):
            self._result = result
            self._exc_info = exc_info

        def result(self, *_):
            if self._exc_info is not None:
                raise self._exc_info[1] from None
            else:
                return self._result

        def exception_info(self, *_):
            return self._exc_info

        def add_done_callback(self, callback):
            callback(self)

    def __init__(self, *_):
        pass

    def submit(self, fn, *args, **kwargs):
        try:
            return self._MockResult(fn(*args, **kwargs))
        except:  # noqa: E722
            return self._MockResult(None, sys.exc_info())

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(None, *args, **kwargs)


class MockExecutorSyncProvider(ThreadExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return MockThreadPoolExecutor(n_workers)


class GraphDeviceAssigner(object):
    # Analyze graph and assign initial chunks to different GPU devices
    # only work when execute on GPU
    def __init__(self, graph, starts, devices):
        self._graph = graph
        self._undigraph = None
        self._op_keys = {start.key for start in starts}
        self._devices = devices
        self._device_slots = {dev: 1 for dev in devices}

    def _calc_device_assign_limits(self, initial_count, occupied=None):
        """
        Calculate limitation of number of initial operands for devices
        :param initial_count: num of nodes in READY state
        :param occupied: device -> num of initials already assigned
        """
        occupied = occupied or dict()
        actual_count = initial_count - sum(occupied.values())

        device_res = sorted(self._device_slots.items(), key=operator.itemgetter(1),
                            reverse=True)

        devices = [t[0] for t in device_res]
        device_cores = np.array([t[1] for t in device_res]).astype(np.float32)

        # remove assigned nodes from limitations
        counts = initial_count * device_cores / device_cores.sum()
        for idx, dev in enumerate(devices):
            counts[idx] = max(0, counts[idx] - occupied.get(dev, 0))

        # all assigned, nothing to do
        if counts.sum() == 0:
            return dict((dev, 0) for dev in devices)

        counts = (actual_count * counts / counts.sum()).astype(np.int32)

        # assign remaining nodes
        pos = 0
        rest = actual_count - counts.sum()
        while rest > 0:
            counts[pos] += 1
            rest -= 1
            pos = (pos + 1) % len(counts)
        return dict(zip(devices, counts))

    def _assign_by_bfs(self, start, device, initial_sizes, spread_limits,
                       keys_to_assign, assigned_record, graph=None):
        """
        Assign initial nodes using Breadth-first Search given initial sizes and
        limitations of spread range.
        """
        if initial_sizes[device] <= 0:
            return

        graph = graph or self._graph
        if self._undigraph is None:
            undigraph = self._undigraph = graph.build_undirected()
        else:
            undigraph = self._undigraph

        assigned = 0
        spread_range = 0
        for v in undigraph.bfs(start=start, visit_predicate='all'):
            op_key = v.op.key
            if op_key in assigned_record:
                continue
            spread_range += 1
            if op_key not in keys_to_assign:
                continue
            assigned_record[op_key] = device
            assigned += 1
            if spread_range >= spread_limits[device] \
                    or assigned >= initial_sizes[device]:
                break
        initial_sizes[device] -= assigned

    def assign(self):
        """
        Decide target device for given chunks.

        :return: dict mapping operand keys into device
        """
        graph = self._graph
        cur_assigns = OrderedDict()

        op_key_to_chunks = defaultdict(list)
        for n in graph:
            op_key_to_chunks[n.op.key].append(n)

        descendant_readies = set()
        op_keys = set(self._op_keys)
        chunks_to_assign = [op_key_to_chunks[k][0] for k in op_keys]
        assigned_counts = defaultdict(lambda: 0)

        # calculate the number of nodes to be assigned to each device
        # given number of devices and existing assignments
        device_quotas = self._calc_device_assign_limits(
            len(chunks_to_assign) + len(descendant_readies), assigned_counts)

        # calculate expected descendant count (spread range) of
        # every device and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self._device_slots)
        spread_ranges = defaultdict(lambda: average_spread_range)
        # assign from other nodes to be assigned
        sorted_candidates = [v for v in chunks_to_assign]
        while max(device_quotas.values()):
            device = max(device_quotas, key=lambda k: device_quotas[k])
            cur = sorted_candidates.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_candidates.pop()
            self._assign_by_bfs(cur, device, device_quotas, spread_ranges, op_keys,
                                cur_assigns, graph=graph)

        keys_to_assign = set(n.op.key for n in chunks_to_assign)
        for k, v in cur_assigns.items():
            if k in keys_to_assign:
                for chunk in op_key_to_chunks[k]:
                    chunk.op._device = v


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

    def handle_op(self, *args, **kw):
        return Executor.handle(*args, **kw)

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
        assigner = GraphDeviceAssigner(self._graph, self._submitted_op_keys, devices)
        assigner.assign()

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

    @enter_mode(kernel=True)
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


class Executor(object):
    _graph_executor_cls = LocalChunkGraphExecutor

    class SyncProviderType(Enum):
        THREAD = 0
        GEVENT = 1
        MOCK = 2

    _sync_provider = {
        SyncProviderType.MOCK: MockExecutorSyncProvider,
        SyncProviderType.THREAD: ThreadExecutorSyncProvider,
        SyncProviderType.GEVENT: GeventExecutorSyncProvider,
    }

    def __init__(self, engine=None, storage=None, prefetch=False,
                 sync_provider_type=SyncProviderType.THREAD):
        self._engine = engine
        self._chunk_result = storage if storage is not None else dict()
        self._prefetch = prefetch

        # dict structure: {tileable_key -> fetch tileables}
        # dict value is a fetch tileable to record metas.
        self.stored_tileables = dict()
        self._tileable_names = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)
        # synchronous provider
        self._sync_provider = self._sync_provider[sync_provider_type]

        self._mock_max_memory = 0

    @property
    def chunk_result(self):
        return self._chunk_result

    @property
    def storage(self):
        return self._chunk_result

    @storage.setter
    def storage(self, new_storage):
        self._chunk_result = new_storage

    @property
    def mock_max_memory(self):
        return self._mock_max_memory

    def execute_graph(self, graph, keys, n_parallel=None, print_progress=False,
                      mock=False, no_intermediate=False, compose=True, retval=True,
                      chunk_result=None):
        """
        :param graph: graph to execute
        :param keys: result keys
        :param n_parallel: num of max parallelism
        :param print_progress:
        :param compose: if True. fuse nodes when possible
        :param mock: if True, only estimate data sizes without execution
        :param no_intermediate: exclude intermediate data sizes when estimating memory size
        :param retval: if True, keys specified in argument keys is returned
        :param chunk_result: dict to put chunk key to chunk data, if None, use self.chunk_result
        :return: execution result
        """
        if compose:
            RuntimeOptimizer(graph, self._engine).optimize(keys=keys)
        optimized_graph = graph

        executed_keys = set()
        for t in itertools.chain(*list(self.stored_tileables.values())):
            executed_keys.update([c.key for c in t.chunks])
        chunk_result = self._chunk_result if chunk_result is None else chunk_result

        if not mock:
            graph_executor = self._graph_executor_cls(
                chunk_result, self._sync_provider, engine=self._engine, n_parallel=n_parallel)
        else:
            graph_executor = MockChunkGraphExecutor(
                chunk_result, self._sync_provider, engine=self._engine, n_parallel=n_parallel,
                mock_max_memory=self._mock_max_memory)
        graph_listener = graph_executor.submit_chunk_graph(optimized_graph, keys)
        graph_listener.wait()
        res = [chunk_result[k] for k in keys] if retval else None

        if mock:
            self._mock_max_memory = max(self._mock_max_memory, graph_executor._mock_max_memory)
            chunk_result.clear()
        return res

    @enter_mode(build=True, kernel=True)
    def execute_tileable(self, tileable, n_parallel=None, n_thread=None, concat=False,
                         print_progress=False, mock=False, compose=True):
        result_keys = []
        tileable_data = tileable.data if hasattr(tileable, 'data') else tileable

        def _on_tile_success(before_tile_data, after_tile_data):
            if before_tile_data is tileable_data:
                if concat and len(after_tile_data.chunks) > 1:
                    after_tile_data = after_tile_data.op.concat_tileable_chunks(after_tile_data)
                result_keys.extend(c.key for c in after_tile_data.chunks)

            return after_tile_data

        # shallow copy
        chunk_result = self._chunk_result.copy()
        tileable_graph_builder = TileableGraphBuilder()
        tileable_graph = tileable_graph_builder.build([tileable])
        chunk_graph_builder = ChunkGraphBuilder(compose=compose,
                                                on_tile_success=_on_tile_success)
        chunk_graph = chunk_graph_builder.build([tileable], tileable_graph=tileable_graph)
        ret = self.execute_graph(chunk_graph, result_keys, n_parallel=n_parallel or n_thread,
                                 print_progress=print_progress, mock=mock,
                                 chunk_result=chunk_result)
        self._chunk_result.update(chunk_result)
        return ret

    execute_tensor = execute_tileable
    execute_dataframe = execute_tileable

    @classmethod
    def _update_chunk_shape(cls, chunk_graph, chunk_result):
        for c in chunk_graph:
            if hasattr(c, 'shape') and c.shape is not None and \
                    any(np.isnan(s) for s in c.shape) and c.key in chunk_result:
                try:
                    c._shape = chunk_result[c.key].shape
                except AttributeError:
                    # Fuse chunk
                    c._composed[-1]._shape = chunk_result[c.key].shape

    def _update_tileable_and_chunk_shape(self, tileable_graph, chunk_result, failed_ops):
        for n in tileable_graph:
            if n.op in failed_ops:
                continue
            tiled_n = get_tiled(n)
            if has_unknown_shape(tiled_n):
                if any(c.key not in chunk_result for c in tiled_n.chunks):
                    # some of the chunks has been fused
                    continue
                new_nsplits = self.get_tileable_nsplits(n, chunk_result=chunk_result)
                for node in (n, tiled_n):
                    node._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
                tiled_n._nsplits = new_nsplits

    @contextlib.contextmanager
    def _gen_local_context(self, chunk_result):
        if isinstance(chunk_result, LocalContext):
            with chunk_result:
                yield chunk_result
        else:
            yield chunk_result

    @enter_mode(build=True, kernel=True)
    def execute_tileables(self, tileables, fetch=True, n_parallel=None, n_thread=None,
                          print_progress=False, mock=False, compose=True, name=None):
        # shallow copy chunk_result, prevent from any chunk key decref
        chunk_result = self._chunk_result.copy()
        tileables = [tileable.data if hasattr(tileable, 'data') else tileable
                     for tileable in tileables]
        tileable_keys = [t.key for t in tileables]
        tileable_keys_set = set(tileable_keys)

        result_keys = []
        to_release_keys = set()
        tileable_data_to_concat_keys = weakref.WeakKeyDictionary()
        tileable_data_to_chunks = weakref.WeakKeyDictionary()

        node_to_fetch = weakref.WeakKeyDictionary()
        skipped_tileables = set()

        def _generate_fetch_tileable(node):
            # Attach chunks to fetch tileables to skip tile.
            if isinstance(node.op, Fetch) and node.key in self.stored_tileables:
                tiled = self.stored_tileables[node.key][0]
                node._chunks = tiled.chunks
                node.nsplits = tiled.nsplits
                for param, v in tiled.params.items():
                    setattr(node, '_' + param, v)

            return node

        def _skip_executed_tileables(inps):
            # skip the input that executed, and not gc collected
            new_inps = []
            for inp in inps:
                if inp.key in self.stored_tileables:
                    try:
                        get_tiled(inp)
                    except KeyError:
                        new_inps.append(inp)
                    else:
                        skipped_tileables.add(inp)
                        continue
                else:
                    new_inps.append(inp)
            return new_inps

        def _generate_fetch_if_executed(nd):
            # node processor that if the node is executed
            # replace it with a fetch node
            _to_fetch = node_to_fetch  # noqa: F821
            if nd.key not in chunk_result:
                return nd
            if nd in _to_fetch:
                return _to_fetch[nd]
            fn = build_fetch(nd).data
            _to_fetch[nd] = fn
            return fn

        def _on_tile_success(before_tile_data, after_tile_data):
            if before_tile_data.key not in tileable_keys_set:
                return after_tile_data
            tile_chunk_keys = [c.key for c in after_tile_data.chunks]
            result_keys.extend(tile_chunk_keys)
            tileable_data_to_chunks[before_tile_data] = [build_fetch(c) for c in after_tile_data.chunks]
            if not fetch:
                pass
            elif len(after_tile_data.chunks) > 1:
                # need to fetch data and chunks more than 1, we concatenate them into 1
                after_tile_data = after_tile_data.op.concat_tileable_chunks(after_tile_data)
                chunk = after_tile_data.chunks[0]
                result_keys.append(chunk.key)
                tileable_data_to_concat_keys[before_tile_data] = chunk.key
                # after return the data to user, we release the reference
                to_release_keys.add(chunk.key)
            else:
                tileable_data_to_concat_keys[before_tile_data] = after_tile_data.chunks[0].key
            return after_tile_data

        def _get_tileable_graph_builder(**kwargs):
            if options.optimize_tileable_graph:
                return OptimizeIntegratedTileableGraphBuilder(**kwargs)
            else:
                return TileableGraphBuilder(**kwargs)

        # As the chunk_result is copied, we cannot use the original context any more,
        # and if `chunk_result` is a LocalContext, it's copied into a LocalContext as well,
        # thus here just to make sure the new context is entered
        with self._gen_local_context(chunk_result):
            # build tileable graph
            tileable_graph_builder = _get_tileable_graph_builder(
                node_processor=_generate_fetch_tileable,
                inputs_selector=_skip_executed_tileables)
            tileable_graph = tileable_graph_builder.build(tileables)
            chunk_graph_builder = IterativeChunkGraphBuilder(
                graph_cls=DAG, node_processor=_generate_fetch_if_executed,
                compose=False, on_tile_success=_on_tile_success)
            intermediate_result_keys = set()
            while True:
                # build chunk graph, tile will be done during building
                chunk_graph = chunk_graph_builder.build(
                    tileables, tileable_graph=tileable_graph)
                tileable_graph = chunk_graph_builder.prev_tileable_graph
                temp_result_keys = set(result_keys)
                if not chunk_graph_builder.done:
                    # add temporary chunks keys into result keys
                    for interrupted_op in chunk_graph_builder.interrupted_ops:
                        for inp in interrupted_op.inputs:
                            if inp.op not in chunk_graph_builder.interrupted_ops:
                                for n in get_tiled(inp).chunks:
                                    temp_result_keys.add(n.key)
                else:
                    # if done, prune chunk graph
                    prune_chunk_graph(chunk_graph, temp_result_keys)
                # compose
                if compose:
                    chunk_graph.compose(list(temp_result_keys))
                # execute chunk graph
                self.execute_graph(chunk_graph, list(temp_result_keys),
                                   n_parallel=n_parallel or n_thread,
                                   print_progress=print_progress, mock=mock,
                                   chunk_result=chunk_result)

                # update shape of tileable and its chunks whatever it's successful or not
                self._update_chunk_shape(chunk_graph, chunk_result)
                self._update_tileable_and_chunk_shape(
                    tileable_graph, chunk_result, chunk_graph_builder.interrupted_ops)

                if chunk_graph_builder.done:
                    if len(intermediate_result_keys) > 0:
                        # failed before
                        intermediate_to_release_keys = \
                            {k for k in intermediate_result_keys
                             if k not in result_keys and k in chunk_result}
                        to_release_keys.update(intermediate_to_release_keys)
                    delattr(chunk_graph_builder, '_prev_tileable_graph')
                    break
                else:
                    intermediate_result_keys.update(temp_result_keys)
                    # add the node that failed
                    to_run_tileables = list(itertools.chain(
                        *(op.outputs for op in chunk_graph_builder.interrupted_ops)))
                    to_run_tileables_set = set(to_run_tileables)
                    for op in chunk_graph_builder.interrupted_ops:
                        for inp in op.inputs:
                            if inp not in to_run_tileables_set:
                                to_run_tileables_set.add(inp)
                    tileable_graph_builder = _get_tileable_graph_builder(
                        inputs_selector=lambda inps: [inp for inp in inps
                                                      if inp in to_run_tileables_set])
                    tileable_graph = tileable_graph_builder.build(to_run_tileables_set)

            if name is not None:
                if not isinstance(name, (list, tuple)):
                    name = [name]
                self._tileable_names.update(zip(name, tileables))

            for tileable in tileables:
                fetch_tileable = build_fetch(get_tiled(tileable, mapping=tileable_optimized))
                fetch_tileable._key = tileable.key
                fetch_tileable._id = tileable.id
                if tileable.key in self.stored_tileables:
                    if tileable.id not in [t.id for t in self.stored_tileables[tileable.key]]:
                        self.stored_tileables[tileable.key].append(fetch_tileable)
                else:
                    self.stored_tileables[tileable.key] = [fetch_tileable]

            try:
                if fetch:
                    concat_keys = [
                        tileable_data_to_concat_keys[tileable_optimized.get(t, t)] for t in tileables]
                    return [chunk_result[k] for k in concat_keys]
                else:
                    return
            finally:
                for to_release_key in to_release_keys:
                    del chunk_result[to_release_key]
                self._chunk_result.update(
                    {k: chunk_result[k] for k in result_keys if k in chunk_result})

    execute_tensors = execute_tileables
    execute_dataframes = execute_tileables

    @classmethod
    def _check_slice_on_tileable(cls, tileable):
        from .tensor.indexing import TensorIndex
        from .dataframe.indexing.iloc import DataFrameIlocGetItem

        if isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem)):
            indexes = tileable.op.indexes
            if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                raise ValueError('Only support fetch data slices')

    @enter_mode(kernel=True)
    def fetch_tileables(self, tileables, **kw):
        from .tensor.indexing import TensorIndex
        from .dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        to_release_tileables = []
        for tileable in tileables:
            if tileable.key not in self.stored_tileables and \
                    isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem)):
                to_fetch_tileable = tileable.inputs[0]
                to_release_tileables.append(tileable)
            else:
                to_fetch_tileable = tileable
            key = to_fetch_tileable.key
            if key not in self.stored_tileables and not isinstance(to_fetch_tileable.op, Fetch):
                # check if the tileable is executed before
                raise ValueError(
                    f'Tileable object {tileable.key} must be executed first before being fetched')

        # if chunk executed, fetch chunk mechanism will be triggered in execute_tileables
        result = self.execute_tileables(tileables, **kw)
        for to_release_tileable in to_release_tileables:
            for c in get_tiled(to_release_tileable, mapping=tileable_optimized).chunks:
                del self._chunk_result[c.key]
        return result

    @classmethod
    def _get_chunk_shape(cls, chunk_key, chunk_result):
        return chunk_result[chunk_key].shape

    def get_tileable_nsplits(self, tileable, chunk_result=None):
        chunk_idx_to_shape = OrderedDict()
        tiled = get_tiled(tileable, mapping=tileable_optimized)
        chunk_result = chunk_result if chunk_result is not None else self._chunk_result
        for chunk in tiled.chunks:
            chunk_idx_to_shape[chunk.index] = self._get_chunk_shape(chunk.key, chunk_result)
        return calc_nsplits(chunk_idx_to_shape)

    def decref(self, *keys):
        rs = set(self._chunk_result)
        for key in keys:
            tileable_key, tileable_id = key
            if tileable_key not in self.stored_tileables:
                continue

            ids = [t.id for t in self.stored_tileables[tileable_key]]
            if tileable_id in ids:
                idx = ids.index(tileable_id)
                tiled = self.stored_tileables[tileable_key][0]
                chunk_keys = set([c.key for c in tiled.chunks])
                self.stored_tileables[tileable_key].pop(idx)
                # for those same key tileables, do decref only when all those tileables are garbage collected
                if len(self.stored_tileables[tileable_key]) != 0:
                    continue
                for chunk_key in (chunk_keys & rs):
                    self._chunk_result.pop(chunk_key, None)
                del self.stored_tileables[tileable_key]


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


# import to register operands
from . import tensor  # noqa: E402
from . import dataframe  # noqa: E402
from . import optimizes  # noqa: E402
from . import learn  # noqa: E402
from . import remote  # noqa: E402

del tensor, dataframe, optimizes, learn, remote
