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

import uuid
import weakref

import numpy as np

from mars.errors import StorageFull
from mars.graph import DAG
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import aio_case, patch_method
from mars.worker import WorkerDaemonActor, DispatchActor, StorageManagerActor, \
    CpuCalcActor, IORunnerActor, PlasmaKeyMapActor, SharedHolderActor, \
    InProcHolderActor, QuotaActor, MemQuotaActor, StatusActor
from mars.worker.storage import DataStorageDevice
from mars.worker.storage.sharedstore import PlasmaSharedStore
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import build_quota_key, WorkerClusterInfoActor


@aio_case
class Test(WorkerCase):
    def _start_calc_pool(self):
        this = self
        mock_addr = '127.0.0.1:%d' % get_next_port()

        class _AsyncContextManager(object):
            async def __aenter__(self):
                self._pool_ctx = this.create_pool(n_process=1, address=mock_addr)
                self._pool = pool = await self._pool_ctx.__aenter__()
                self._test_actor_ctx = this.run_actor_test(pool)
                test_actor = await self._test_actor_ctx.__aenter__()

                await pool.create_actor(SchedulerClusterInfoActor, [mock_addr],
                                        uid=SchedulerClusterInfoActor.default_uid())
                await pool.create_actor(WorkerClusterInfoActor, [mock_addr],
                                        uid=WorkerClusterInfoActor.default_uid())

                await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
                await pool.create_actor(StatusActor, mock_addr, uid=StatusActor.default_uid())

                await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
                await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
                await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
                await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
                await pool.create_actor(IORunnerActor)
                await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
                await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())
                await pool.create_actor(InProcHolderActor)
                await pool.create_actor(CpuCalcActor, uid=CpuCalcActor.default_uid())

                return pool, test_actor

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self._pool.actor_ref(SharedHolderActor.default_uid()).destroy()
                await self._test_actor_ctx.__aexit__(exc_type, exc_val, exc_tb)
                await self._pool.__aexit__(exc_type, exc_val, exc_tb)

        return _AsyncContextManager()

    @staticmethod
    def _build_test_graph(data_list):
        from mars.tensor.fetch import TensorFetch
        from mars.tensor.arithmetic import TensorTreeAdd

        inputs = []
        for idx, d in enumerate(data_list):
            chunk_key = 'chunk-%d' % idx
            fetch_chunk = TensorFetch(to_fetch_key=chunk_key, dtype=d.dtype) \
                .new_chunk([], shape=d.shape, _key=chunk_key)
            inputs.append(fetch_chunk)
        add_chunk = TensorTreeAdd(data_list[0].dtype).new_chunk(inputs, shape=data_list[0].shape)

        exec_graph = DAG()
        exec_graph.add_node(add_chunk)
        for input_chunk in inputs:
            exec_graph.add_node(input_chunk)
            exec_graph.add_edge(input_chunk, add_chunk)
        return exec_graph, inputs, add_chunk

    async def testCpuCalcSingleFetches(self):
        async with self._start_calc_pool() as (_pool, test_actor):
            quota_ref = test_actor.promise_ref(MemQuotaActor.default_uid())
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(3)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                await self.waitp(
                    await storage_client.put_objects(
                        session_id, [fetch_chunk.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )
            self.assertEqual(list((await storage_client.get_data_locations(session_id, [fetch_chunks[0].key]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            quota_batch = {
                build_quota_key(session_id, add_chunk.key, add_chunk.op.key): data_list[0].nbytes,
            }

            for idx in [1, 2]:
                quota_batch[build_quota_key(session_id, fetch_chunks[idx].key, add_chunk.op.key)] \
                    = data_list[idx].nbytes

                await self.waitp(
                    (await storage_client.copy_to(session_id, [fetch_chunks[idx].key], [DataStorageDevice.DISK]))
                        .then(lambda *_: storage_client.delete(
                            session_id, [fetch_chunks[idx].key], [DataStorageDevice.SHARED_MEMORY]))
                )
                self.assertEqual(
                    list((await storage_client.get_data_locations(session_id, [fetch_chunks[idx].key]))[0]),
                    [(0, DataStorageDevice.DISK)])

            await self.waitp(
                quota_ref.request_batch_quota(quota_batch, _promise=True),
            )

            o_create = PlasmaSharedStore.create

            def _mock_plasma_create(store, session_id, data_key, size):
                if data_key == fetch_chunks[2].key:
                    raise StorageFull
                return o_create(store, session_id, data_key, size)

            ref_store = []

            async def _extract_value_ref(*_):
                inproc_handler = await storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))
                obj = (await inproc_handler.get_objects(session_id, [add_chunk.key]))[0]
                ref_store.append(weakref.ref(obj))
                del obj

            with patch_method(PlasmaSharedStore.create, _mock_plasma_create):
                await self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(_extract_value_ref)
                        .then(lambda *_: calc_ref.store_results(
                            session_id, add_chunk.op.key, [add_chunk.key], None, _promise=True))
                )

            self.assertIsNone(ref_store[-1]())

            quota_dump = await quota_ref.dump_data()
            self.assertEqual(len(quota_dump.allocations), 0)
            self.assertEqual(len(quota_dump.requests), 0)
            self.assertEqual(len(quota_dump.proc_sizes), 0)
            self.assertEqual(len(quota_dump.hold_sizes), 0)

            self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [fetch_chunks[0].key]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [fetch_chunks[1].key]))[0]),
                             [(0, DataStorageDevice.DISK)])
            self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [fetch_chunks[2].key]))[0]),
                             [(0, DataStorageDevice.DISK)])
            self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [add_chunk.key]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

    async def testCpuCalcErrorInRunning(self):
        async with self._start_calc_pool() as (_pool, test_actor):
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(2)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                await self.waitp(
                    await storage_client.put_objects(
                        session_id, [fetch_chunk.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )

            def _mock_calc_results_error(*_, **__):
                raise ValueError

            with patch_method(CpuCalcActor._calc_results, _mock_calc_results_error), \
                    self.assertRaises(ValueError):
                await self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(lambda *_: calc_ref.store_results(
                            session_id, add_chunk.op.key, [add_chunk.key], None, _promise=True))
                )
