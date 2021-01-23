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
import operator
import os
import sys
import tempfile
import time
import unittest
import uuid
from functools import reduce

import numpy as np
import pandas as pd

import mars.dataframe as md
import mars.tensor as mt
from mars.actors import new_client
from mars.errors import ExecutionFailed
from mars.scheduler.custom_log import CustomLogMetaActor
from mars.scheduler.resource import ResourceActor
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest
from mars.scheduler.utils import JobState
from mars.scheduler.tests.integrated.no_prepare_op import PureDependsOperand
from mars.session import new_session
from mars.utils import serialize_graph
from mars.remote import spawn, ExecutableTuple
from mars.tests.core import EtcdProcessHelper, require_cupy, require_cudf
from mars.context import DistributedContext

logger = logging.getLogger(__name__)


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(SchedulerIntegratedTest):
    def testMainTensorWithoutEtcd(self):
        self.start_processes(n_schedulers=1)
        sess = new_session(self.session_manager_ref.address)

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()

        actor_client = new_client()
        job_id = str(uuid.uuid4())
        sess_ref = actor_client.actor_ref(self.session_manager_ref.get_session_refs()[sess.session_id])
        sess_ref.submit_job(serialize_graph(c.build_graph()), job_id, [c.key])

        while sess_ref.get_job_status(job_id) != JobState.SUCCEEDED:
            time.sleep(1)
