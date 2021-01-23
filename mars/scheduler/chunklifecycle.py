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

import logging
import os
from collections import defaultdict

from .utils import SchedulerActor

logger = logging.getLogger(__name__)


class ChunkLifecycleActor(SchedulerActor):
    @staticmethod
    def gen_uid(session_id):
        return f's:0:chunk_lifecycle@{session_id}'

    def __init__(self, session_id):
        super().__init__()
        self._session_id = session_id
        self._ref_counts = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d at %s', self.uid, os.getpid(), self.address)
        super().post_create()
        self.set_cluster_info_ref()

    def incref(self, keys):
        for k in keys:
            try:
                self._ref_counts[k] += 1
            except KeyError:
                self._ref_counts[k] = 1

    def decref(self, keys):
        keys_to_delete = []
        for k in keys:
            self._ref_counts[k] -= 1
            if self._ref_counts[k] == 0:
                del self._ref_counts[k]
                keys_to_delete.append(k)
        self._batch_delete_keys(keys)

    def _batch_delete_keys(self, keys):
        from ..worker.execution import ExecutionActor

        workers_list = self.chunk_meta.batch_get_workers(self._session_id, keys)
        worker_to_keys = defaultdict(list)
        for k, workers in zip(keys, workers_list):
            if not workers:
                continue
            for worker in workers:
                worker_to_keys[worker].append(k)

        for worker, keys in worker_to_keys.items():
            worker_ref = self.ctx.actor_ref(ExecutionActor.default_uid(), address=worker)
            worker_ref.delete_data_by_keys.async_tell(self._session_id, keys)

    def remove_if_no_ref(self, keys):
        keys_to_delete = []
        for k in keys:
            if k not in self._ref_counts:
                keys_to_delete.append(k)
        self._batch_delete_keys(keys_to_delete)
