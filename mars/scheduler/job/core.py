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

from ...graph import DAG
from ...tiles import ChunkGraphBuilder
from ..utils import SchedulerActor, JobState


class JobActor(SchedulerActor):
    @staticmethod
    def gen_uid(session_id, job_id):
        return f's:0:job${job_id}@{session_id}'

    def __init__(self, session_id, job_id, tileable_graph, target_tileables):
        super().__init__()
        self._session_id = session_id
        self._job_id = job_id

        self._tileable_graph = DAG()
        tileable_graph.copyto(self._tileable_graph)
        self._target_tileables = set(target_tileables)

        self._chunk_graph = None
        self._chunk_target_op_keys = set()

        self._session_ref = None
        self._chunk_lifecycle_ref = None

        self._last_execution = None

    def post_create(self):
        super().post_create()
        self.set_cluster_info_ref()

        from ..session import SessionActor
        self._session_ref = self.ctx.actor_ref(SessionActor.gen_uid(self._session_id))

        from ..chunklifecycle import ChunkLifecycleActor
        self._chunk_lifecycle_ref = self.get_actor_ref(ChunkLifecycleActor.gen_uid(self._session_id))

    def start_execution(self):
        from .execution import JobExecution

        self._session_ref.update_job_status(self._job_id, JobState.PREPARING, _tell=True)

        if self._chunk_graph is None:
            chunk_graph_builder = ChunkGraphBuilder(
                graph=None, graph_cls=DAG, compose=True)
            self._chunk_graph = chunk_graph_builder.build([self], tileable_graph=self._tileable_graph)

        for t in self._tileable_graph.iter_indep(reverse=True):
            self._chunk_target_op_keys.update(c.op.key for c in t.chunks)
        self._chunk_lifecycle_ref.incref(self._chunk_target_op_keys)

        self._last_execution = JobExecution(
            self._session_id, self._job_id, self._chunk_graph, actor_ctx=self.ctx,
            job_address=self.address)

        self._session_ref.update_job_status(self._job_id, JobState.RUNNING, _tell=True)
        self._last_execution.submit_all()

    def set_operand_worker(self, op_key, worker, input_metas):
        self._last_execution.submit_operand_to_worker(op_key, worker, input_metas)

    def set_operand_finished(self, op_key):
        self._last_execution.set_operand_finished(op_key)
        self._chunk_target_op_keys.difference_update([op_key])
        if not self._chunk_target_op_keys:
            self._last_execution.clean_all()
            self._last_execution = None
            self._session_ref.update_job_status(self._job_id, JobState.SUCCEEDED, _tell=True)
