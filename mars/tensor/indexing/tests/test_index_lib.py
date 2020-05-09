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

import asyncio
import unittest

import numpy as np

from mars.tensor import tensor
from mars.tensor.indexing.index_lib import NDArrayIndexesHandler
from mars.tests.core import ExecutorForTest, aio_case
from mars.tiles import get_tiled


@aio_case
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ExecutorForTest('numpy')

    def testAggregateResult(self):
        rs = np.random.RandomState(0)
        raw = rs.rand(10, 10)
        t = tensor(raw, chunk_size=6)

        slc = slice(None, None, 3)

        # test no reorder
        fancy_index = np.array([3, 6, 7])
        indexes = [slc, fancy_index]
        result = t[indexes].tiles()

        handler = NDArrayIndexesHandler()
        chunk_results = None

        async def test_exec():
            nonlocal chunk_results
            context = await handler.handle(result.op, return_context=True)
            self.assertGreater(context.op.outputs[0].chunk_shape[-1], 1)
            chunk_results = await self.executor.execute_tensor(result, _async=True)
            chunk_results = \
                [(c.index, r) for c, r in zip(get_tiled(result).chunks, chunk_results)]
            expected = (await self.executor.execute_tensor(result, concat=True, _async=True))[0]
            res = handler.aggregate_result(context, chunk_results)
            np.testing.assert_array_equal(res, expected)

        asyncio.get_event_loop().run_until_complete(test_exec())

        # test fancy index that requires reordering
        fancy_index = np.array([6, 7, 3])
        indexes = [slc, fancy_index]
        test = t[indexes].tiles()

        async def test_exec():
            nonlocal chunk_results
            context = await handler.handle(test.op, return_context=True)
            self.assertEqual(context.op.outputs[0].chunk_shape[-1], 1)
            res = handler.aggregate_result(context, chunk_results)
            expected = (await self.executor.execute_tensor(test, concat=True, _async=True))[0]
            np.testing.assert_array_equal(res, expected)

        asyncio.get_event_loop().run_until_complete(test_exec())
