# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
import queue
import random
import time
from collections import defaultdict

from ..errors import WorkerDead
from ..utils import log_unhandled
from .taskheap import TaskHeap
from .utils import SchedulerActor, rewrite_worker_errors

logger = logging.getLogger(__name__)


class AssignerActor(SchedulerActor):
    """
    Actor handling worker assignment queries from operands
    and returning appropriate workers.
    """
    def __init__(self, session_id):
        super(AssignerActor, self).__init__()
        self._session_id = session_id

        self._worker_metrics = None
        # since worker metrics does not change frequently, we update it
        # only when it is out of date
        self._worker_metric_time = 0

        self._initial_heap = TaskHeap()
        self._task_heap = TaskHeap()

        self._resource_ref = None
        self._cluster_info_ref = None
        self._assigner_service_ref = None

        self._notified_workers = set()

    @staticmethod
    def gen_uid(session_id):
        return 's:h1:assigner$%s' % session_id

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        from .resource import ResourceActor
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        try:
            service_uid = self.ctx.distributor.make_same_process(
                's:h1:assign_service$%s' % self._session_id, self.uid)
        except AttributeError:
            service_uid = 's:h1:assign_service$%s' % self._session_id
        self._assigner_service_ref = self.ctx.create_actor(
            AssignerServiceActor, self.ref(), uid=service_uid)

    def pre_destroy(self):
        super(AssignerActor, self).pre_destroy()

    def _refresh_worker_metrics(self):
        def _adjust_heap_size(heap):
            workers = set(self._worker_metrics.keys())
            heap_workers = set(heap.groups)
            removed = heap_workers - workers
            for w in removed:
                heap.remove_group(w)
            self._notified_workers.difference_update(removed)

            for w in workers - heap_workers:
                heap.add_group(w)

        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < t:
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t

            _adjust_heap_size(self._initial_heap)
            _adjust_heap_size(self._task_heap)

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    def filter_alive_workers(self, workers, refresh=False):
        if refresh:
            self._refresh_worker_metrics()
        return [w for w in workers if w in self._worker_metrics] if self._worker_metrics else []

    @log_unhandled
    def submit_initial(self, op_key, priority_data, target_worker):
        assert target_worker is not None

        self._refresh_worker_metrics()
        self._initial_heap.add_task(
            (self._session_id, op_key), priority_data, [target_worker])
        self._notify_workers([target_worker])

    @log_unhandled
    def submit_operand(self, op_key, executable_graph, op_io_meta, priority_data,
                       key_to_metas, target_worker=None):
        """
        Register resource request for an operand
        """
        self._refresh_worker_metrics()

        data_sizes = dict((k, meta.chunk_size) for k, meta in key_to_metas.items())
        chunk_workers = dict((k, meta.workers) for k, meta in key_to_metas.items())

        # already assigned valid target worker, return directly
        if target_worker and target_worker in self._worker_metrics:
            candidate_workers = [target_worker]
        else:
            candidate_workers = self._get_eps_by_worker_locality(
                op_io_meta['input_chunks'], chunk_workers, data_sizes)

        self._task_heap.add_task(
            (self._session_id, op_key), priority_data, candidate_workers,
            self._session_id, op_key, executable_graph, op_io_meta, data_sizes
        )
        logger.debug('Operand %s enqueued, candidate workers are %r',
                     op_key, candidate_workers)

        self._notify_workers(candidate_workers)
        return candidate_workers

    def update_operand_priority(self, session_id, op_key, priority):
        try:
            self._task_heap.update_priority((session_id, op_key), priority)
            return
        except KeyError:
            pass
        try:
            self._initial_heap.update_priority((session_id, op_key), priority)
        except KeyError:
            pass

    def pop_worker_initial(self, worker):
        try:
            return self._initial_heap.pop_group_task(worker)
        except queue.Empty:
            return None

    def pop_worker_task(self, worker):
        try:
            return self._task_heap.pop_group_task(worker)
        except queue.Empty:
            return None

    def unregister_worker_notify(self, worker):
        self._notified_workers.difference_update([worker])

    def _get_eps_by_worker_locality(self, input_keys, chunk_workers, input_sizes):
        locality_data = defaultdict(lambda: 0)
        for k in input_keys:
            if k in chunk_workers:
                for ep in chunk_workers[k]:
                    locality_data[ep] += input_sizes[k]
        workers = list(self._worker_metrics.keys())
        random.shuffle(workers)
        max_locality = -1
        max_eps = []
        for ep in workers:
            if locality_data[ep] > max_locality:
                max_locality = locality_data[ep]
                max_eps = [ep]
            elif locality_data[ep] == max_locality:
                max_eps.append(ep)
        return max_eps

    def _notify_workers(self, workers):
        from ..worker.taskqueue import TaskQueueActor

        notified = self._notified_workers
        worker_futures = []
        for w in workers:
            if w in notified:
                continue
            ref = self.ctx.actor_ref(TaskQueueActor.default_uid(), address=w)
            worker_futures.append(
                (w, ref.register_task_source(self._assigner_service_ref, _tell=True, _wait=False))
            )
            logger.debug('Notification for tasks on worker %s registered', w)
            notified.add(w)

        dead_workers = []
        for w, f in worker_futures:
            try:
                with rewrite_worker_errors():
                    f.result()
            except WorkerDead:
                dead_workers.append(w)
        if dead_workers:
            self._resource_ref.detach_dead_workers(dead_workers, _tell=True)


class AssignerServiceActor(SchedulerActor):
    def __init__(self, assigner_ref):
        super(AssignerServiceActor, self).__init__()
        self._assigner_ref = assigner_ref

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(AssignerServiceActor, self).post_create()
        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(self._assigner_ref)

    def pop_worker_tasks(self, worker, task_num, already_popped):
        from .operands import OperandActor

        task_items = []
        for _ in range(task_num):
            item = self._assigner_ref.pop_worker_task(worker)
            if item is not None:
                logger.debug('Operand %s assigned to worker %s', item.key[-1], worker)
                task_items.append(item)
            else:
                break
        if not task_items:
            logger.debug('Tasks of worker %s exhausted, will not notify any more', worker)
            self._assigner_ref.unregister_worker_notify(worker)

        pop_futures = []
        for _ in range(max(0, task_num - len(task_items) - already_popped)):
            item = self._assigner_ref.pop_worker_initial(worker)
            if item is None:
                break
            op_uid = OperandActor.gen_uid(*item.key)
            pop_futures.append(self.get_actor_ref(op_uid).submit_operand(_tell=True, _wait=False))

        [f.result() for f in pop_futures]

        return task_items, len(pop_futures)
