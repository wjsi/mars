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
import time

from ..errors import WorkerDead
from .taskheap import TaskHeap
from .utils import SchedulerActor, rewrite_worker_errors

logger = logging.getLogger(__name__)


class TaskPoolActor(SchedulerActor):
    @staticmethod
    def gen_uid(session_id):
        return 's:h1:task_pool$%s' % session_id

    def __init__(self, session_id):
        super(TaskPoolActor, self).__init__()
        self._session_id = session_id

        self._resource_ref = None
        self._task_heap = TaskHeap()
        self._worker_metric_time = 0
        self._workers = None
        self._notified_workers = set()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(TaskPoolActor, self).post_create()
        self.set_cluster_info_ref()

        from .resource import ResourceActor
        self._resource_ref = self.get_actor_ref(ResourceActor.default_name())

    def _refresh_worker_metrics(self):
        t = time.time()
        if self._workers is None or self._worker_metric_time + 1 < t:
            # update worker metrics from ResourceActor
            workers = self._workers = set(self._resource_ref.get_workers_meta())
            self._worker_metric_time = t

            heap_workers = set(self._task_heap.workers)
            removed = heap_workers - workers
            for w in removed:
                self._task_heap.remove_worker(w)
            self._notified_workers.difference_update(removed)

            for w in workers - heap_workers:
                self._task_heap.add_worker(w)

    def _notify_workers(self, workers):
        from ..worker.taskqueue import TaskQueueActor

        notified = self._notified_workers
        worker_futures = []
        for w in workers:
            if w in notified:
                continue
            ref = self.ctx.actor_ref(TaskQueueActor.default_name(), address=w)
            worker_futures.append(
                (w, ref.register_task_source(self.ref(), _tell=True, _wait=False))
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

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    def submit_task(self, session_id, op_key, priority, workers, *args, **kwargs):
        logger.debug('Task %s added into task heap, candidate workers are %r', op_key, workers)
        self._refresh_worker_metrics()
        self._task_heap.add_task((session_id, op_key), priority, list(workers), *args, **kwargs)
        self._notify_workers(workers)

    def remove_task(self, session_id, op_key):
        self._task_heap.remove_task((session_id, op_key))

    def update_task_priorities(self, session_id, op_key, priority):
        try:
            self._task_heap.update_priority((session_id, op_key), priority)
        except KeyError:
            pass

    def pop_worker_tasks(self, worker, task_num):
        task_items = []
        for _ in range(task_num):
            try:
                task_items.append(self._task_heap.pop_worker_task(worker))
            except queue.Empty:
                break
        if not task_items:
            logger.debug('Tasks of worker %s exhausted, will not notify any more', worker)
            self._notified_workers.difference_update([worker])
        return task_items
