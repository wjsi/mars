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

import sys
import time
import unittest

import pandas as pd
import numpy as np

from mars.lib.adjustable_tpe import AdjustableThreadPoolExecutor
from mars.lib.groupby_wrapper import GroupByWrapper, wrapped_groupby
from mars.tests.core import assert_groupby_equal
from mars.utils import calc_data_size


class Test(unittest.TestCase):
    def testGroupByWrapper(self):
        df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                                 'foo', 'bar', 'foo', 'foo'],
                           'B': ['one', 'one', 'two', 'three',
                                 'two', 'two', 'one', 'three'],
                           'C': np.random.randn(8),
                           'D': np.random.randn(8)},
                          index=pd.MultiIndex.from_tuples([(i // 4, i) for i in range(8)]))

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, level=0).to_tuple())
        assert_groupby_equal(grouped, df.groupby(level=0))
        self.assertEqual(grouped.shape, (8, 4))
        self.assertTrue(grouped.is_frame)
        self.assertGreater(sys.getsizeof(grouped), sys.getsizeof(grouped.groupby_obj))
        self.assertGreater(calc_data_size(grouped), sys.getsizeof(grouped.groupby_obj))

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, level=0).C.to_tuple())
        assert_groupby_equal(grouped, df.groupby(level=0).C)
        self.assertEqual(grouped.shape, (8,))
        self.assertFalse(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, 'B').to_tuple())
        assert_groupby_equal(grouped, df.groupby('B'))
        self.assertEqual(grouped.shape, (8, 4))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, 'B').C.to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby('B').C, with_selection=True)
        self.assertEqual(grouped.shape, (8,))
        self.assertFalse(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, 'B')[['C', 'D']].to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby('B')[['C', 'D']], with_selection=True)
        self.assertEqual(grouped.shape, (8, 2))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df, ['B', 'C']).to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby(['B', 'C']))
        self.assertEqual(grouped.shape, (8, 4))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, ['B', 'C']).C.to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby(['B', 'C']).C, with_selection=True)
        self.assertEqual(grouped.shape, (8,))
        self.assertFalse(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, ['B', 'C'])[['A', 'D']].to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby(['B', 'C'])[['A', 'D']], with_selection=True)
        self.assertEqual(grouped.shape, (8, 2))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, ['B', 'C'])[['C', 'D']].to_tuple(truncate=True))
        assert_groupby_equal(grouped, df.groupby(['B', 'C'])[['C', 'D']], with_selection=True)
        self.assertEqual(grouped.shape, (8, 2))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, lambda x: x[-1] % 2).to_tuple(pickle_function=True))
        assert_groupby_equal(grouped, df.groupby(lambda x: x[-1] % 2), with_selection=True)
        self.assertEqual(grouped.shape, (8, 4))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, lambda x: x[-1] % 2).C.to_tuple(pickle_function=True))
        assert_groupby_equal(grouped, df.groupby(lambda x: x[-1] % 2).C, with_selection=True)
        self.assertEqual(grouped.shape, (8,))
        self.assertFalse(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(
            wrapped_groupby(df, lambda x: x[-1] % 2)[['C', 'D']].to_tuple(pickle_function=True))
        assert_groupby_equal(grouped, df.groupby(lambda x: x[-1] % 2)[['C', 'D']], with_selection=True)
        self.assertEqual(grouped.shape, (8, 2))
        self.assertTrue(grouped.is_frame)

        grouped = GroupByWrapper.from_tuple(wrapped_groupby(df.B, lambda x: x[-1] % 2).to_tuple())
        assert_groupby_equal(grouped, df.B.groupby(lambda x: x[-1] % 2), with_selection=True)
        self.assertEqual(grouped.shape, (8,))
        self.assertFalse(grouped.is_frame)

    def testAdjustablePool(self):
        delay_time = 1.0
        delay_eps = delay_time / 4
        start_times = dict()
        end_times = dict()
        futures = []

        def work_fun(idx):
            start_times[idx] = time.time()
            time.sleep(delay_time)
            end_times[idx] = time.time()

        pool = AdjustableThreadPoolExecutor(1)

        t1 = time.time()
        for idx in range(4):
            futures.append(pool.submit(work_fun, idx))

        time.sleep(delay_time / 2)
        t2 = time.time()

        pool.increase_workers()
        time.sleep(delay_eps)
        pool.decrease_workers()
        t3 = time.time()
        self.assertEqual(len(pool._threads), 1)

        pool.increase_workers()
        for f in futures:
            f.result()
        t4 = time.time()
        self.assertEqual(len(pool._threads), 2)

        pool.decrease_workers()
        t5 = time.time()
        self.assertEqual(len(pool._threads), 1)

        self.assertGreater(t2 - t1, delay_time / 2 - delay_eps)
        self.assertLess(t2 - t1, delay_time / 2 + delay_eps)
        self.assertGreater(t3 - t2, delay_time / 2 - delay_eps)
        self.assertLess(t3 - t2, delay_time / 2 + delay_eps)
        self.assertGreater(t4 - t3, delay_time * 3 / 2 - delay_eps)
        self.assertLess(t4 - t3, delay_time * 3 / 2 + delay_eps)
        self.assertLess(t5 - t4, delay_time / 2)

        for idx in range(3):
            self.assertGreater(start_times[idx + 1] - start_times[idx], delay_time / 2 - delay_eps)
            self.assertLess(start_times[idx + 1] - start_times[idx], delay_time / 2 + delay_eps)

            self.assertGreater(end_times[idx + 1] - end_times[idx], delay_time / 2 - delay_eps)
            self.assertLess(end_times[idx + 1] - end_times[idx], delay_time / 2 + delay_eps)
