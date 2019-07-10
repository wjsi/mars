# distutils: language = c++
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

from collections import namedtuple
from queue import Empty

cimport cython
from libcpp.vector cimport vector
from cpython.object cimport Py_LT, Py_GT, PyObject_RichCompareBool

cdef Py_ssize_t _INVALID_GROUP_POS = 0x8fffffff


_D_TaskStoreItem = namedtuple('_D_TaskStoreItem', 'positions qids key priority groups args kwargs')
_D_TaskHeapItem = namedtuple('_D_TaskHeapItem', 'store_item position_idx')
_D_TaskHeap = namedtuple('_D_TaskHeapItem', 'queues free_queues group_to_queues store_items')


cdef class TaskStoreItem:
    """
    Operand start info with priority
    """
    # vector recording indices of queues the item belongs to
    cdef vector[Py_ssize_t] qids
    # vector recording position of the item in queues
    cdef vector[Py_ssize_t] positions

    cpdef public object key
    cpdef public object priority
    cpdef public object groups
    cpdef public object args
    cpdef public object kwargs

    def __init__(self, object key, object priority, object groups, object args, object kwargs):
        self.key = key
        self.priority = priority
        self.groups = groups
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return '<%s key=%r priority=%r>' % (type(self).__name__, self.key, self.priority)

    def dump(self):
        return _D_TaskStoreItem(
            list(self.positions), list(self.qids), self.key, self.priority, self.groups, self.args, self.kwargs
        )

    def __reduce__(self):
        return TaskStoreItem, (self.key, self.priority, self.groups, self.args, self.kwargs)


cdef class TaskHeapItem:
    """
    Data item in the heap
    """
    cdef TaskStoreItem store_item
    # index of the queue in OpStoreItem.positions and OpStoreItem.qids
    cdef int position_idx

    def __init__(self, TaskStoreItem item, int idx):
        self.store_item = item
        self.position_idx = idx

    def __repr__(self):
        return '<%s store_item=%r position_idx=%r>' % \
               (type(self).__name__, self.store_item, self.position_idx)

    def dump(self):
        return _D_TaskHeapItem(self.store_item.dump(), self.position_idx)


cdef class GroupDefItem:
    cdef object key
    cdef bint max_heap

    def __init__(self, object key, bint max_heap=True):
        self.key = key
        self.max_heap = max_heap


cdef class TaskHeapGroups:
    """
    Heaps to record tasks for operands in groups.
    """
    cdef list _group_defs
    cdef list _queues
    cdef list _free_queues
    # mapping from groups to queue indices
    cdef dict _group_to_queues

    # mapping from data keys to instances of OpStoreItem
    cdef dict _store_items

    def __init__(self):
        self._group_defs = list()
        self._queues = list()
        self._free_queues = list()
        self._group_to_queues = dict()
        self._store_items = dict()

    def dump(self):
        return _D_TaskHeap(self._queues, self._free_queues, self._group_to_queues, self._store_items)

    @property
    def groups(self):
        return list(self._group_to_queues)

    def __len__(self):
        return len(self._store_items)

    def __getitem__(self, item):
        return self._store_items[item]

    cpdef void add_group(self, object group_key, bint max_heap=True):
        cdef Py_ssize_t free_idx = 0
        cdef Py_ssize_t queue_idx = 0
        cdef Py_ssize_t n_free_queues = len(self._free_queues)
        cdef GroupDefItem group_def = GroupDefItem(group_key, max_heap)

        if free_idx < n_free_queues:
            # vacant position available, put directly
            queue_idx = self._free_queues[free_idx]
            self._queues[queue_idx] = []
            self._group_defs[queue_idx] = group_def
            free_idx += 1
        else:
            # create new position
            queue_idx = len(self._queues)
            self._queues.append([])
            self._group_defs.append(group_def)

        self._group_to_queues[group_key] = queue_idx
        if free_idx > 0:
            self._free_queues = self._free_queues[free_idx:]

    cpdef void remove_group(self, object group_key):
        cdef TaskHeapItem hitem
        cdef Py_ssize_t qid = self._group_to_queues[group_key]
        cdef list group_queue = self._queues[qid]

        for hitem in group_queue:
            # mark one queue as invalid
            # not removing to ensure OpHeapItem.position_idx still goes right
            hitem.store_item.qids[hitem.position_idx] = _INVALID_GROUP_POS
            if hitem.store_item.positions.size() == 1:
                # no other queues, can delete safely
                del self._store_items[hitem.store_item.key]

        del group_queue[:]
        self._free_queues.append(qid)
        del self._group_to_queues[group_key]
        self._group_defs[qid] = None

    def add_task(self, object key, object priority, list group_keys, *args, **kwargs):
        cdef TaskStoreItem store_item = TaskStoreItem(key, priority, group_keys, args, kwargs)
        self.add_task_item(store_item)

    def add_task_item(self, TaskStoreItem store_item):
        cdef Py_ssize_t qid, pos, i, j, free_id
        cdef Py_ssize_t group_num = len(store_item.groups)
        cdef list group_queue

        try:
            # delete first in case the key already exists
            self.remove_task(store_item.key)
        except KeyError:
            pass

        self._store_items[store_item.key] = store_item
        store_item.positions.reserve(group_num)
        store_item.qids.reserve(group_num)

        for i, w in enumerate(store_item.groups):
            qid = self._group_to_queues[w]
            group_queue = self._queues[qid]
            pos = len(group_queue)
            group_queue.append(TaskHeapItem(store_item, i))
            store_item.positions.push_back(pos)
            store_item.qids.push_back(qid)
            self._sift_up(qid, pos)

    cpdef remove_task(self, object key):
        cdef size_t i
        cdef Py_ssize_t qid, pos, qsize
        cdef TaskHeapItem last_item
        cdef TaskStoreItem store_item = self._store_items[key]
        cdef list queue

        # remove task in all queues
        for i in range(store_item.positions.size()):
            qid = store_item.qids[i]
            if qid == _INVALID_GROUP_POS:
                continue
            pos = store_item.positions[i]

            queue = self._queues[qid]
            qsize = len(queue)

            last_item = queue[pos] = queue[-1]
            # we should keep positions correctly
            last_item.store_item.positions[last_item.position_idx] = pos
            # now the original item can be safely removed
            queue.pop(-1)

            if pos < qsize - 1 and qsize - 1 != 0:
                self._sift_up(qid, pos)
                self._sift_down(qid, pos)

        del self._store_items[store_item.key]

    cpdef object pop_group_task(self, object group_key):
        cdef list queue = self._queues[self._group_to_queues[group_key]]

        if len(queue) == 0:
            raise Empty

        cdef TaskHeapItem hitem = queue[0]
        cdef TaskStoreItem store_item = hitem.store_item
        self.remove_task(store_item.key)
        return store_item

    cpdef update_priority(self, object key, object priority):
        cdef size_t i
        cdef TaskStoreItem store_item = self._store_items[key]
        cdef TaskHeapItem heap_item

        store_item.priority = priority

        # adjust every queue containing the element
        for i in range(store_item.positions.size()):
            qid = store_item.qids[i]
            if qid == _INVALID_GROUP_POS:
                continue
            pos = store_item.positions[i]
            self._sift_up(qid, pos)
            self._sift_down(qid, pos)

    @staticmethod
    @cython.nonecheck(False)
    cdef inline int _compare_heap_items(TaskHeapItem o1, TaskHeapItem o2, int op_cmp) except -1:
        cdef int cmp_result = PyObject_RichCompareBool(
            o1.store_item.priority, o2.store_item.priority, op_cmp)
        if cmp_result < 0:
            raise TypeError('Cannot compare %r with %r with op %d' % (o1, o2, op_cmp))
        return cmp_result

    cdef void _sift_up(self, Py_ssize_t qid, Py_ssize_t pos) except *:
        cdef TaskHeapItem cur_item, par_item
        cdef GroupDefItem group_def = self._group_defs[qid]
        cdef list queue = self._queues[qid]
        cdef Py_ssize_t ppos = (pos - 1) >> 1
        cdef int op_cmp = Py_GT if group_def.max_heap else Py_LT

        cur_item = queue[pos]

        while pos > 0:
            par_item = queue[ppos]
            if TaskHeapGroups._compare_heap_items(cur_item, par_item, op_cmp):
                # move smaller parent into current pos and update position record
                queue[pos] = par_item
                par_item.store_item.positions[par_item.position_idx] = pos
            else:
                # place handling data into current pos and update position record
                queue[pos] = cur_item
                cur_item.store_item.positions[cur_item.position_idx] = pos
                return
            pos = ppos
            ppos = (ppos - 1) >> 1

        # we move to the top of the heap, just place
        queue[0] = cur_item
        cur_item.store_item.positions[cur_item.position_idx] = 0

    cdef void _sift_down(self, Py_ssize_t qid, Py_ssize_t pos) except *:
        cdef TaskHeapItem cur_item, child_item, child_right_item, par_item
        cdef GroupDefItem group_def = self._group_defs[qid]
        cdef list queue = self._queues[qid]
        cdef Py_ssize_t cpos = (pos << 1) + 1, maxpos = len(queue)
        # the first leaf node
        cdef Py_ssize_t limit = maxpos >> 1
        cdef int op_cmp = Py_LT if group_def.max_heap else Py_GT

        cur_item = queue[pos]

        while pos < limit:
            child_item = queue[cpos]
            if cpos + 1 < maxpos:
                # if the right sibling has higher priority, use it
                child_right_item = queue[cpos + 1]
                if TaskHeapGroups._compare_heap_items(child_item, child_right_item, op_cmp):
                    child_item = child_right_item
                    cpos += 1

            if TaskHeapGroups._compare_heap_items(cur_item, child_item, op_cmp):
                # move maximal child to parent
                queue[pos] = child_item
                child_item.store_item.positions[child_item.position_idx] = pos
            else:
                # place handling data on the parent node as no further move needed
                queue[pos] = cur_item
                cur_item.store_item.positions[cur_item.position_idx] = pos
                return
            pos = cpos
            cpos = (cpos << 1) + 1

        # we move to the leaves of the heap, just place
        queue[pos] = cur_item
        cur_item.store_item.positions[cur_item.position_idx] = pos


__all__ = ['TaskHeapGroups', 'TaskStoreItem', 'Empty']
