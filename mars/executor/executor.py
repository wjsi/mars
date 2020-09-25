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
import weakref
import contextlib
from collections import defaultdict, OrderedDict
from numbers import Integral

import numpy as np

from ..config import options
from ..graph import DAG
from ..operands import Fetch
from ..tiles import IterativeChunkGraphBuilder, ChunkGraphBuilder, get_tiled
from ..optimizes.runtime.core import RuntimeOptimizer
from ..optimizes.tileable_graph import tileable_optimized, OptimizeIntegratedTileableGraphBuilder
from ..graph_builder import TileableGraphBuilder
from ..context import LocalContext
from ..utils import enter_mode, build_fetch, calc_nsplits,\
    has_unknown_shape, prune_chunk_graph
from .core import SyncProviderType, get_sync_provider
from .chunk_executor import LocalChunkGraphExecutor, MockChunkGraphExecutor

logger = logging.getLogger(__name__)


class Executor(object):
    _graph_executor_cls = LocalChunkGraphExecutor

    def __init__(self, engine=None, storage=None, prefetch=False, n_parallel=None,
                 sync_provider_type=SyncProviderType.THREAD, mock=False):
        from ..session import Session

        self._engine = engine
        self._chunk_result = storage if storage is not None else LocalContext(Session.default_or_local())
        self._prefetch = prefetch

        # dict structure: {tileable_key -> fetch tileables}
        # dict value is a fetch tileable to record metas.
        self.stored_tileables = dict()
        self._tileable_names = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)
        # synchronous provider
        self._sync_provider = get_sync_provider(sync_provider_type)

        self._mock = mock
        self._mock_max_memory = 0

        if not mock:
            self._graph_executor = self._graph_executor_cls(
                self._chunk_result, self._sync_provider, engine=self._engine, n_parallel=n_parallel)
        else:
            self._graph_executor = MockChunkGraphExecutor(
                self._chunk_result, self._sync_provider, engine=self._engine, n_parallel=n_parallel,
                mock_max_memory=self._mock_max_memory)

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

        graph_listener = self._graph_executor.submit_chunk_graph(optimized_graph, keys)
        graph_listener.wait()
        res = [self._chunk_result[k] for k in keys] if retval else None

        if self._mock:
            self._mock_max_memory = max(self._mock_max_memory, self._graph_executor._mock_max_memory)
            self._chunk_result.clear()
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
        tileable_graph_builder = TileableGraphBuilder()
        tileable_graph = tileable_graph_builder.build([tileable])
        chunk_graph_builder = ChunkGraphBuilder(compose=compose,
                                                on_tile_success=_on_tile_success)
        chunk_graph = chunk_graph_builder.build([tileable], tileable_graph=tileable_graph)
        ret = self.execute_graph(chunk_graph, result_keys, n_parallel=n_parallel or n_thread,
                                 print_progress=print_progress)
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
                    try:
                        c._composed[-1]._shape = chunk_result[c.key].shape
                    except AttributeError:
                        pass

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
        tileables = [tileable.data if hasattr(tileable, 'data') else tileable
                     for tileable in tileables]
        tileable_keys = [t.key for t in tileables]
        tileable_keys_set = set(tileable_keys)

        result_keys = []
        to_release_keys = set()
        tileable_data_to_concat_keys = weakref.WeakKeyDictionary()
        tileable_data_to_chunks = weakref.WeakKeyDictionary()

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
        with self._gen_local_context(self._chunk_result):
            # build tileable graph
            tileable_graph_builder = _get_tileable_graph_builder(
                node_processor=_generate_fetch_tileable,
                inputs_selector=_skip_executed_tileables)
            tileable_graph = tileable_graph_builder.build(tileables)
            chunk_graph_builder = IterativeChunkGraphBuilder(
                graph_cls=DAG, compose=False, on_tile_success=_on_tile_success)
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
                                   print_progress=print_progress)

                # update shape of tileable and its chunks whatever it's successful or not
                self._update_chunk_shape(chunk_graph, self._chunk_result)
                self._update_tileable_and_chunk_shape(
                    tileable_graph, self._chunk_result, chunk_graph_builder.interrupted_ops)

                if chunk_graph_builder.done:
                    if len(intermediate_result_keys) > 0:
                        # failed before
                        intermediate_to_release_keys = \
                            {k for k in intermediate_result_keys
                             if k not in result_keys and k in self._chunk_result}
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
                    return [self._chunk_result[k] for k in concat_keys]
                else:
                    return
            finally:
                for to_release_key in to_release_keys:
                    del self._chunk_result[to_release_key]
                self._chunk_result.update(
                    {k: self._chunk_result[k] for k in result_keys if k in self._chunk_result})

    execute_tensors = execute_tileables
    execute_dataframes = execute_tileables

    @classmethod
    def _check_slice_on_tileable(cls, tileable):
        from ..tensor.indexing import TensorIndex
        from ..dataframe.indexing.iloc import DataFrameIlocGetItem

        if isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem)):
            indexes = tileable.op.indexes
            if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                raise ValueError('Only support fetch data slices')

    @enter_mode(kernel=True)
    def fetch_tileables(self, tileables, **kw):
        from ..tensor.indexing import TensorIndex
        from ..dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

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

    def increase_pool_size(self):
        self._graph_executor.increase_pool_size()

    def decrease_pool_size(self):
        self._graph_executor.decrease_pool_size()
