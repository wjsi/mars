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

import json
import logging
import operator
import os
import sys
import unittest
import uuid
from functools import reduce

import numpy as np
from numpy.testing import assert_allclose

from mars import tensor as mt
from mars.actors.core import new_client
from mars.scheduler.graph import GraphState
from mars.scheduler.resource import ResourceActor
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest
from mars.scheduler.tests.integrated.no_prepare_op import NoPrepareOperand
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.utils import build_tileable_graph
from mars.serialize.dataserializer import loads
from mars.tests.core import EtcdProcessHelper, require_cupy, require_cudf, aio_case
from mars.context import DistributedContext

logger = logging.getLogger(__name__)


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
@aio_case
class Test(SchedulerIntegratedTest):
    async def testMainTensorWithoutEtcd(self):
        await self.start_processes()

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        logger.warning('Test Case 0')
        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        graph = await c.build_graph(_async=True)
        targets = [c.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, c.key)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        assert_allclose(loads(result), expected.sum())

        logger.warning('Test Case 1')
        a = mt.ones((100, 50), chunk_size=35) * 2 + 1
        b = mt.ones((50, 200), chunk_size=35) * 2 + 1
        c = a.dot(b)
        graph = await c.build_graph(_async=True)
        targets = [c.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)
        result = await session_ref.fetch_result(graph_key, c.key)
        assert_allclose(loads(result), np.ones((100, 200)) * 450)

        logger.warning('Test Case 2')
        base_arr = np.random.random((100, 100))
        a = mt.array(base_arr)
        sumv = reduce(operator.add, [a[:10, :10] for _ in range(10)])
        graph = await sumv.build_graph(_async=True)
        targets = [sumv.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        expected = reduce(operator.add, [base_arr[:10, :10] for _ in range(10)])
        result = await session_ref.fetch_result(graph_key, sumv.key)
        assert_allclose(loads(result), expected)

        logger.warning('Test Case 3')
        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        r = b.sum(axis=1)
        graph = await r.build_graph(_async=True)
        targets = [r.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, r.key)
        assert_allclose(loads(result), np.ones((27, 31)).sum(axis=1))

        logger.warning('Test Case 4')
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=(5, 4))
        b = a[a.argmin(axis=1), mt.tensor(np.arange(10))]
        graph = await b.build_graph(_async=True)
        targets = [b.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, b.key)

        np.testing.assert_array_equal(loads(result), raw[raw.argmin(axis=1), np.arange(10)])

    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    async def testMainTensorWithEtcd(self):
        await self.start_processes(etcd=True)

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        graph = await c.build_graph(_async=True)
        targets = [c.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, c.key)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        assert_allclose(loads(result), expected.sum())

    @require_cupy
    @require_cudf
    async def testMainTensorWithCuda(self):
        await self.start_processes(cuda=True)

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        a = mt.ones((100, 100), chunk_size=30, gpu=True) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30, gpu=True) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        graph = await c.build_graph(_async=True)
        targets = [c.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, c.key)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        assert_allclose(loads(result), expected.sum())

    async def testMainDataFrameWithoutEtcd(self):
        import pandas as pd
        from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
        from mars.dataframe.datasource.series import from_pandas as from_pandas_series
        from mars.dataframe.arithmetic import add

        await self.start_processes(etcd=False, scheduler_args=['-Dscheduler.aggressive_assign=true'])

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas_df(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas_df(data2, chunk_size=6)

        df3 = add(df1, df2)

        graph = await df3.build_graph(_async=True)
        targets = [df3.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        expected = data1 + data2
        result = await session_ref.fetch_result(graph_key, df3.key)
        pd.testing.assert_frame_equal(expected, loads(result))

        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas_df(data1, chunk_size=(10, 5))
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas_df(data2, chunk_size=(10, 6))

        df3 = add(df1, df2)

        graph = await df3.build_graph(_async=True)
        targets = [df3.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        expected = data1 + data2
        result = await session_ref.fetch_result(graph_key, df3.key)
        pd.testing.assert_frame_equal(expected, loads(result))

        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas_df(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas_df(data2, chunk_size=6)

        df3 = add(df1, df2)

        graph = await df3.build_graph(_async=True)
        targets = [df3.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        expected = data1 + data2
        result = await session_ref.fetch_result(graph_key, df3.key)
        pd.testing.assert_frame_equal(expected, loads(result))

        s1 = pd.Series(np.random.rand(10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        series1 = from_pandas_series(s1)

        graph = await series1.build_graph(_async=True)
        targets = [series1.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, series1.key)
        pd.testing.assert_series_equal(s1, loads(result))

    async def testIterativeTilingWithoutEtcd(self):
        await self.start_processes(etcd=False)

        session_id = uuid.uuid1()
        actor_client = new_client()
        rs = np.random.RandomState(0)

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        raw = rs.rand(100)
        a = mt.tensor(raw, chunk_size=10)
        a.sort()
        c = a[:5]

        graph = await c.build_graph(_async=True)
        targets = [c.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, c.key)
        expected = np.sort(raw)[:5]
        assert_allclose(loads(result), expected)

        with self.assertRaises(KeyError):
            await session_ref.fetch_result(graph_key, a.key, check=False)

        raw1 = rs.rand(20)
        raw2 = rs.rand(20)
        a = mt.tensor(raw1, chunk_size=10)
        a.sort()
        b = mt.tensor(raw2, chunk_size=15) + 1
        c = mt.concatenate([a[:10], b])
        c.sort()
        d = c[:5]

        graph = await d.build_graph(_async=True)
        targets = [d.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = await session_ref.fetch_result(graph_key, d.key)
        expected = np.sort(np.concatenate([np.sort(raw1)[:10], raw2 + 1]))[:5]
        assert_allclose(loads(result), expected)

        raw = rs.randint(100, size=(100,))
        a = mt.tensor(raw, chunk_size=53)
        a.sort()
        b = mt.histogram(a, bins='scott')

        graph = build_tileable_graph(b, set())
        targets = [b[0].key, b[1].key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        res = await session_ref.fetch_result(graph_key, b[0].key), \
            await session_ref.fetch_result(graph_key, b[1].key)
        expected = np.histogram(np.sort(raw), bins='scott')
        assert_allclose(loads(res[0]), expected[0])
        assert_allclose(loads(res[1]), expected[1])

    async def testDistributedContext(self):
        await self.start_processes(etcd=False)

        session_id = uuid.uuid1()
        actor_client = new_client()
        rs = np.random.RandomState(0)

        cluster_info_ref = actor_client.actor_ref(SchedulerClusterInfoActor.default_uid(),
                                                  address=self.scheduler_endpoints[0])
        context = DistributedContext(
            scheduler_address=self.scheduler_endpoints[0], session_id=session_id,
            is_distributed=await cluster_info_ref.is_distributed())

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))
        raw1 = rs.rand(10, 10)
        a = mt.tensor(raw1, chunk_size=4)

        graph = await a.build_graph(_async=True)
        targets = [a.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()), graph_key,
                                                target_tileables=targets, names=['test'])

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        tileable_key = await context.get_tileable_key_by_name('test')
        self.assertEqual(a.key, tileable_key)

        nsplits = (await context.get_tileable_metas([a.key], filter_fields=['nsplits']))[0][0]
        self.assertEqual(((4, 4, 2), (4, 4, 2)), nsplits)

        r = await context.get_tileable_data(a.key)
        np.testing.assert_array_equal(raw1, r)

        indexes = [slice(3, 9), slice(0, 7)]
        r = await context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[tuple(indexes)], r)

        indexes = [[1, 4, 2, 4, 5], slice(None, None, None)]
        r = await context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[tuple(indexes)], r)

        indexes = ([9, 1, 2, 0], [0, 0, 4, 4])
        r = await context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[[9, 1, 2, 0], [0, 0, 4, 4]], r)

    async def testOperandsWithoutPrepareInputs(self):
        await self.start_processes(etcd=False, modules=['mars.scheduler.tests.integrated.no_prepare_op'])

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(await self.session_manager_ref.create_session(session_id))

        actor_address = await self.cluster_info.get_scheduler(ResourceActor.default_uid())
        resource_ref = actor_client.actor_ref(ResourceActor.default_uid(), address=actor_address)
        worker_endpoints = await resource_ref.get_worker_endpoints()

        t1 = mt.random.rand(10)
        t1.op._expect_worker = worker_endpoints[0]
        t2 = mt.random.rand(10)
        t2.op._expect_worker = worker_endpoints[1]

        t = NoPrepareOperand().new_tileable([t1, t2])
        t.op._prepare_inputs = [False, False]

        graph = await t.build_graph(_async=True)
        targets = [t.key]
        graph_key = uuid.uuid1()
        await session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                                graph_key, target_tileables=targets)

        state = await self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)
