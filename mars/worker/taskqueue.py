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

import heapq
import logging
import time

from .utils import WorkerActor
from .. import resource
from ..utils import log_unhandled

logger = logging.getLogger(__name__)
_ALLOCATE_PERIOD = 0.5


class ReverseWrapper(object):
    __slots__ = '_item',

    def __init__(self, item):
        self._item = item

    def __lt__(self, other):
        return self._item > other._item


class TaskQueueActor(WorkerActor):
    """
    Actor accepting requests and holding the queue
    """
    def __init__(self, parallel_num=None):
        super(TaskQueueActor, self).__init__()
        self._allocated = set()
        self._req_heap = []

        self._allocator_ref = None
        self._parallel_num = parallel_num or resource.cpu_count()

        self._last_load_source = 0
        self._task_sources = []
        self._popped_lengths = []
        self._empty_sources = []

    def post_create(self):
        super(TaskQueueActor, self).post_create()

        self._allocator_ref = self.ctx.create_actor(
            TaskQueueAllocatorActor, self.ref(), self._parallel_num,
            uid=TaskQueueAllocatorActor.default_uid())

    def register_task_source(self, ref):
        ref = self.ctx.actor_ref(ref)
        if self._empty_sources:
            pos = self._empty_sources.pop(-1)
            self._task_sources[pos] = ref
            self._popped_lengths[pos] = 0
        else:
            self._task_sources.append(ref)
            self._popped_lengths.append(0)

        self.ref().load_tasks(_tell=True)

    def release_task(self, session_id, op_key):
        """
        Remove an operand task from queue
        :param session_id: session id
        :param op_key: operand key
        """
        logger.debug('Operand task %s released in %s.', op_key, self.address)
        query_key = (session_id, op_key)
        try:
            self._allocated.remove(query_key)
        except KeyError:
            pass

        # as one task has been released, we can perform allocation again
        self._allocator_ref.enable_quota(_tell=True)
        self._allocator_ref.allocate_tasks(_tell=True)

    def load_tasks(self):
        if not self._task_sources:
            return

        load_num = self._parallel_num \
            - max(int(0.5 + resource.cpu_percent() / 100), len(self._allocated)) \
            - len(self._req_heap)
        n_sources = len(self._task_sources)
        start_src = cur_src = self._last_load_source
        loaded_count = 0

        while load_num > 0:
            ref = self._task_sources[cur_src]
            if ref is not None:
                tasks, self._popped_lengths[cur_src] = ref.pop_worker_tasks(
                    self.address, load_num, self._popped_lengths[cur_src])
                load_num -= len(tasks)
                if tasks:
                    loaded_count += len(tasks)
                    req_heap = self._req_heap
                    for task in tasks:
                        heapq.heappush(
                            req_heap, (ReverseWrapper((task.priority, id(task))), task))
                else:
                    self._task_sources[cur_src] = None
                    self._empty_sources.append(cur_src)
            cur_src = (cur_src + 1) % n_sources
            if cur_src == start_src:
                break

        self._last_load_source = cur_src
        if loaded_count:
            self._allocator_ref.allocate_tasks(_tell=True)

    def mark_allocated(self, key):
        """
        Mark an operand as being allocated, i.e., it has been submitted to the MemQuotaActor.
        :param key: operand key
        """
        self._allocated.add(key)

    def get_allocated_count(self):
        """
        Get total number of operands allocated to run and already running
        """
        return len(self._allocated)

    def pop_next_request(self):
        """
        Get next unscheduled item from queue. If nothing found, None will
        be returned
        """
        item = None
        if self._req_heap:
            item = heapq.heappop(self._req_heap)
        return item[-1] if item is not None else None


class TaskQueueAllocatorActor(WorkerActor):
    """
    Actor performing periodical assignment
    """
    def __init__(self, queue_ref, parallel_num):
        super(TaskQueueAllocatorActor, self).__init__()
        self._parallel_num = parallel_num
        self._has_quota = True

        self._queue_ref = queue_ref
        self._mem_quota_ref = None
        self._execution_ref = None
        self._last_memory_available = 0
        self._last_allocate_time = time.time() - 2

    def post_create(self):
        super(TaskQueueAllocatorActor, self).post_create()

        from .quota import MemQuotaActor
        from .execution import ExecutionActor

        self._queue_ref = self.ctx.actor_ref(self._queue_ref)
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())
        self._execution_ref = self.ctx.actor_ref(ExecutionActor.default_uid())

        self.ref().allocate_tasks(periodical=True, _delay=_ALLOCATE_PERIOD, _tell=True)

    def enable_quota(self):
        self._has_quota = True

    @log_unhandled
    def allocate_tasks(self, periodical=False):
        # make sure the allocation period is not too dense
        if periodical and self._last_allocate_time > time.time() - _ALLOCATE_PERIOD:
            return
        cur_mem_available = resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            self._has_quota = True
        self._last_memory_available = cur_mem_available

        num_cpu = resource.cpu_count()
        cpu_rate = resource.cpu_percent()
        batch_allocated = 0
        while self._has_quota:
            allocated_count = self._queue_ref.get_allocated_count()
            if allocated_count >= self._parallel_num:
                break
            if allocated_count >= num_cpu / 4 and num_cpu * 100 - 50 < cpu_rate + batch_allocated * 100:
                break
            if self._mem_quota_ref.has_pending_requests():
                break

            item = self._queue_ref.pop_next_request()
            if item is None:
                self._queue_ref.load_tasks(_tell=True)
                break

            # actually submit graph to execute
            # here we do not use _tell=True, as we need to make sure that
            # quota request is sent
            self._execution_ref.execute_graph(*item.args, **item.kwargs)
            self._queue_ref.mark_allocated(item.key)
            batch_allocated += 1
            self.ctx.sleep(0.001)

        self._last_allocate_time = time.time()
        self.ref().allocate_tasks(periodical=True, _delay=_ALLOCATE_PERIOD, _tell=True)
