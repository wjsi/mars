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

import copy
import heapq
import logging
import os
import random
import sys
import time
from collections import defaultdict

from .. import promise
from ..compat import PY27
from ..config import options
from ..errors import DependencyMissing
from ..utils import log_unhandled
from .operands import BaseOperandActor
from .resource import ResourceActor
from .utils import SchedulerActor

logger = logging.getLogger(__name__)


class ChunkPriorityItem(object):
    __slots__ = '_op_key', '_session_id', '_op_info', '_target_worker', '_callback', '_priority'

    """
    Class providing an order for operands for assignment
    """
    def __init__(self, session_id, op_key, op_info, callback, priority=None):
        self._op_key = op_key
        self._session_id = session_id
        self._op_info = op_info
        self._target_worker = op_info.get('target_worker')
        self._callback = callback

        if priority:
            self._priority = priority
        else:
            self._priority = ()
            self.update_priority(op_info['optimize'])

    def update_priority(self, priority_data, copyobj=False):
        obj = self
        if copyobj:
            obj = copy.deepcopy(obj)

        priorities = []
        priorities.extend([
            priority_data.get('depth', 0),
            priority_data.get('demand_depths', ()),
            -priority_data.get('successor_size', 0),
            -priority_data.get('placement_order', 0),
            priority_data.get('descendant_size'),
        ])
        obj._priority = tuple(priorities)
        return obj

    @property
    def session_id(self):
        return self._session_id

    @property
    def op_key(self):
        return self._op_key

    @property
    def target_worker(self):
        return self._target_worker

    @target_worker.setter
    def target_worker(self, value):
        self._target_worker = value

    @property
    def callback(self):
        return self._callback

    @property
    def op_info(self):
        return self._op_info

    if PY27:
        def __reduce__(self):
            return type(self), (self._session_id, self._op_key, self._op_info, self._callback,
                                self._priority)

    def __repr__(self):
        return '<ChunkPriorityItem(%s(%s))>' % (self.op_key, self.op_info['op_name'])

    def __lt__(self, other):
        return self._priority > other._priority


class ReversedHeapItem(ChunkPriorityItem):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], ChunkPriorityItem):
            for k in self.__slots__:
                try:
                    setattr(self, k, getattr(args[0], k))
                except AttributeError:
                    pass
        else:
            super(ReversedHeapItem, self).__init__(*args, **kwargs)

    def __lt__(self, other):
        return self._priority < other._priority


class NotStealableError(Exception):
    pass


class AssignerActor(SchedulerActor):
    """
    Actor handling worker assignment requests from operands.
    Note that this actor does not assign workers itself.
    """
    @staticmethod
    def gen_uid(session_id):
        return 's:h1:assigner$%s' % session_id

    def __init__(self):
        super(AssignerActor, self).__init__()
        self._requests = dict()
        self._req_heap = []
        self._steal_heap = []

        self._cluster_info_ref = None
        self._actual_ref = None
        self._resource_ref = None

        self._worker_metrics = None
        # since worker metrics does not change frequently, we update it
        # only when it is out of date
        self._worker_metric_time = 0
        self._stealer_workers = set()

        self._avg_net_speed = 0
        self._avg_calc_speeds = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        session_id = self.uid.rsplit('$', 1)[-1]
        self._actual_ref = self.ctx.create_actor(AssignEvaluationActor, self.ref(),
                                                 uid=AssignEvaluationActor.gen_uid(session_id))
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

    def pre_destroy(self):
        self._actual_ref.destroy()

    def allocate_top_resources(self):
        self._actual_ref.allocate_top_resources(_tell=True)

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0
        self._actual_ref.mark_metrics_expired(_tell=True)

    def _refresh_worker_metrics(self):
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t

            # collect stealers
            for ep, metric in self._worker_metrics.items():
                if metric.get('last_busy_time', t) < t - options.scheduler.stealer_idle_time:
                    self._stealer_workers.add(ep)
                else:
                    self._stealer_workers.difference_update([ep])

            # calc average transfer speed
            min_stats = options.scheduler.steal_min_data_count
            total_speed, total_count = 0, 0
            for ep in self._worker_metrics:
                net_speed_stats = self._worker_metrics[ep].get('stats', {}).get('net_transfer_speed', {})
                ep_count = net_speed_stats.get('count', 0)
                total_speed += net_speed_stats.get('mean', 0)
                total_count += ep_count

            if total_count >= min_stats:
                self._avg_net_speed = total_speed / len(self._worker_metrics)

            # calc average calc speed
            collected_ops = set()
            for ep in self._worker_metrics:
                for stat_key, calc_stats in self._worker_metrics[ep].get('stats', {}).items():
                    if stat_key.startswith('calc_speed.'):
                        collected_ops.add(stat_key.split('.', 1)[-1])
            for op_name in collected_ops:
                total_speed, total_count = 0, 0
                for ep in self._worker_metrics:
                    calc_stats = self._worker_metrics[ep].get('stats', {}).get('calc_speed.' + op_name, {})
                    ep_count = calc_stats.get('count', 0)
                    total_speed += calc_stats.get('mean', 0)
                    total_count += ep_count

                if total_count < min_stats:
                    continue
                self._avg_calc_speeds[op_name] = total_speed / total_count

            self._actual_ref.update_steal_data(
                self._stealer_workers, self._avg_net_speed, self._avg_calc_speeds, _tell=True, _wait=False)

    def _is_op_roughly_stealable(self, op_info):
        op_name = op_info['op_name']
        if not options.scheduler.enable_worker_steal or not op_name or op_name not in self._avg_calc_speeds:
            return False
        total_data_count = len(op_info['io_meta']['predecessors']) + len(op_info['io_meta']['successors'])
        scaled_speed = self._avg_net_speed / total_data_count * options.scheduler.stealable_threshold
        if scaled_speed < self._avg_calc_speeds[op_name]:
            return True

    def filter_alive_workers(self, workers, refresh=False):
        if refresh:
            self._refresh_worker_metrics()
        return [w for w in workers if w in self._worker_metrics] if self._worker_metrics else []

    def _enqueue_operand(self, session_id, op_key, op_info, callback=None):
        priority_item = ChunkPriorityItem(session_id, op_key, op_info, callback)
        if priority_item.target_worker not in self._worker_metrics:
            priority_item.target_worker = None
        self._requests[op_key] = priority_item
        heapq.heappush(self._req_heap, priority_item)
        if self._stealer_workers and self._is_op_roughly_stealable(op_info):
            heapq.heappush(self._steal_heap, ReversedHeapItem(priority_item))

    @promise.reject_on_exception
    @log_unhandled
    def apply_for_resource(self, session_id, op_key, op_info, callback=None):
        """
        Register resource request for an operand
        :param session_id: session id
        :param op_key: operand key
        :param op_info: operand information, should be a dict
        :param callback: promise callback, called when the resource is assigned
        """
        self._refresh_worker_metrics()
        self._enqueue_operand(session_id, op_key, op_info, callback)
        logger.debug('Operand %s enqueued', op_key)
        self._actual_ref.allocate_top_resources(_tell=True)

    @log_unhandled
    def apply_for_multiple_resources(self, session_id, applications):
        self._refresh_worker_metrics()
        logger.debug('%d operands applied for session %s', len(applications), session_id)
        for app in applications:
            op_key, op_info = app
            self._enqueue_operand(session_id, op_key, op_info)
        self._actual_ref.allocate_top_resources(_tell=True)

    @log_unhandled
    def update_priority(self, op_key, priority_data):
        """
        Update priority data for an operand. The priority item will be
        pushed into priority queue again.
        :param op_key: operand key
        :param priority_data: new priority data
        """
        if op_key not in self._requests:
            return
        obj = self._requests[op_key].update_priority(priority_data, copyobj=True)
        heapq.heappush(self._req_heap, obj)
        if self._stealer_workers and self._is_op_roughly_stealable(obj.op_info):
            heapq.heappush(self._steal_heap, ReversedHeapItem(obj))

    @log_unhandled
    def remove_apply(self, op_key):
        """
        Cancel request for an operand
        :param op_key: operand key
        """
        if op_key in self._requests:
            del self._requests[op_key]

    def _pop_heap_item(self, h):
        item = None
        while h:
            item = heapq.heappop(h)
            if item.op_key in self._requests:
                # use latest request item
                item = self._requests[item.op_key]
                break
            else:
                item = None
        return item

    def pop_head_request(self):
        """
        Pop and obtain top-priority request from queue
        :return: top item
        """
        return self._pop_heap_item(self._req_heap)

    def pop_head_steal(self):
        return self._pop_heap_item(self._steal_heap)

    def extend_requests(self, items):
        """
        Extend heap by an iterable object. The heap will be reheapified.
        :param items: priority items
        """
        self._req_heap.extend(items)
        heapq.heapify(self._req_heap)

    def extend_steal(self, items):
        """
        Extend heap by an iterable object. The heap will be reheapified.
        :param items: priority items
        """
        self._req_heap.extend(items)
        heapq.heapify(self._req_heap)


class AssignEvaluationActor(SchedulerActor):
    """
    Actor assigning operands to workers
    """
    @classmethod
    def gen_uid(cls, session_id):
        return 's:0:%s$%s' % (cls.__name__, session_id)

    def __init__(self, assigner_ref):
        super(AssignEvaluationActor, self).__init__()
        self._worker_metrics = None
        self._worker_metric_time = time.time() - 2

        self._cluster_info_ref = None
        self._assigner_ref = assigner_ref
        self._resource_ref = None

        self._stealer_workers = set()

        self._avg_net_speed = 0
        self._avg_calc_speeds = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(self._assigner_ref)
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        self.periodical_allocate()

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    def periodical_allocate(self):
        self.allocate_top_resources()
        self.ref().periodical_allocate(_tell=True, _delay=0.5)

    def update_steal_data(self, stealers, avg_net_speed, avg_calc_speeds):
        self._stealer_workers = stealers
        self._avg_net_speed = avg_net_speed
        self._avg_calc_speeds = avg_calc_speeds

    def allocate_top_resources(self):
        """
        Allocate resources given the order in AssignerActor
        """
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t
        if not self._worker_metrics:
            return

        unassigned = []
        reject_alloc_workers = set()
        meta_cache = dict()
        # the assigning procedure will continue till
        while len(reject_alloc_workers) < len(self._worker_metrics) - len(self._stealer_workers):
            item = self._assigner_ref.pop_head_request()
            if not item:
                break
            try:
                alloc_ep, rejects = self._allocate_resource(
                    item.session_id, item.op_key, item.op_info, item.target_worker,
                    reject_workers=reject_alloc_workers, meta_cache=meta_cache,
                    use_worker_stats=False)
            except:  # noqa: E722
                logger.exception('Unexpected error occurred in %s', self.uid)
                if item.callback:  # pragma: no branch
                    self.tell_promise(item.callback, *sys.exc_info(), **dict(_accept=False))
                continue

            # collect workers failed to assign operand to
            reject_alloc_workers.update(rejects)
            if alloc_ep:
                # assign successfully, we remove the application
                self._assigner_ref.remove_apply(item.op_key)
            else:
                # put the unassigned item into unassigned list to add back to the queue later
                unassigned.append(item)
        if unassigned:
            # put unassigned back to the queue, if any
            self._assigner_ref.extend_requests(unassigned)

        unassigned = []
        reject_steal_workers = set()
        while len(reject_steal_workers) < len(self._stealer_workers):
            item = self._assigner_ref.pop_head_steal()
            if not item:
                break
            try:
                alloc_ep, rejects = self._allocate_resource(
                    item.session_id, item.op_key, item.op_info, item.target_worker,
                    reject_workers=reject_alloc_workers, meta_cache=meta_cache,
                    use_worker_stats=True)
            except NotStealableError:
                # do nothing
                continue
            except:  # noqa: E722
                logger.exception('Unexpected error occurred in %s', self.uid)
                if item.callback:  # pragma: no branch
                    self.tell_promise(item.callback, *sys.exc_info(), **dict(_accept=False))
                continue

            # collect workers failed to assign operand to
            reject_steal_workers.update(rejects)
            if alloc_ep:
                # assign successfully, we remove the application
                logger.debug('Operand %s stolen in %s', item.op_key, alloc_ep)
                self._assigner_ref.remove_apply(item.op_key)
            else:
                # put the unassigned item into unassigned list to add back to the queue later
                unassigned.append(item)

        if unassigned:
            # put unassigned back to the queue, if any
            self._assigner_ref.extend_steal(unassigned)

    @log_unhandled
    def _allocate_resource(self, session_id, op_key, op_info, target_worker=None, reject_workers=None,
                           use_worker_stats=False, meta_cache=None):
        """
        Allocate resource for single operand
        :param session_id: session id
        :param op_key: operand key
        :param op_info: operand info dict
        :param target_worker: worker to allocate, can be None
        :param reject_workers: workers denied to assign to
        """
        if target_worker not in self._worker_metrics:
            target_worker = None

        reject_workers = reject_workers or set()

        op_io_meta = op_info.get('io_meta', {})
        try:
            input_metas = op_io_meta['input_data_metas']
            input_data_keys = list(input_metas.keys())
            input_sizes = dict((k, v.chunk_size) for k, v in input_metas.items())
        except KeyError:
            input_data_keys = op_io_meta.get('input_chunks', {})

            input_metas = self._get_chunks_meta(session_id, input_data_keys, meta_cache)
            if any(m is None for m in input_metas.values()):
                raise DependencyMissing('Dependency missing for operand %s' % op_key)

            input_sizes = dict((k, meta.chunk_size) for k, meta in input_metas.items())

        if target_worker is None:
            who_has = dict((k, meta.workers) for k, meta in input_metas.items())
            if use_worker_stats:
                output_size = op_io_meta['total_output_size'] or sum(input_sizes.values())
                candidate_workers = self._get_eps_by_worker_stats(
                    input_data_keys, who_has, input_sizes, output_size, op_info['op_name'])
                if candidate_workers is None:
                    raise NotStealableError
            else:
                candidate_workers = self._get_eps_by_worker_locality(input_data_keys, who_has, input_sizes)
        else:
            candidate_workers = [target_worker]

        candidate_workers = [w for w in candidate_workers if w not in reject_workers]
        if not candidate_workers:
            return None, []

        # todo make more detailed allocation plans
        calc_device = op_info.get('calc_device', 'cpu')
        if calc_device == 'cpu':
            alloc_dict = dict(cpu=options.scheduler.default_cpu_usage, memory=sum(input_sizes.values()))
        elif calc_device == 'cuda':
            alloc_dict = dict(cuda=options.scheduler.default_cuda_usage, memory=sum(input_sizes.values()))
        else:  # pragma: no cover
            raise NotImplementedError('Calc device %s not supported.' % calc_device)

        rejects = []
        for worker_ep in candidate_workers:
            if self._resource_ref.allocate_resource(session_id, op_key, worker_ep, alloc_dict):
                logger.debug('Operand %s(%s) allocated to run in %s', op_key, op_info['op_name'], worker_ep)

                self.get_actor_ref(BaseOperandActor.gen_uid(session_id, op_key)) \
                    .submit_to_worker(worker_ep, input_metas, _tell=True, _wait=False)
                return worker_ep, rejects
            rejects.append(worker_ep)
        return None, rejects

    def _get_chunks_meta(self, session_id, keys, meta_cache=None):
        if not keys:
            return dict()
        result = dict()
        miss_keys = []
        for k in keys:
            try:
                result[k] = meta_cache[k]
            except KeyError:
                miss_keys.append(k)
        result.update(zip(miss_keys, self.chunk_meta.batch_get_chunk_meta(session_id, miss_keys)))
        return result

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

    def _get_eps_by_worker_stats(self, input_keys, who_has, input_sizes, output_size, op_name):
        ep_transmit_times = defaultdict(list)
        for key in input_keys:
            contain_eps = who_has.get(key, set())
            for ep in self._stealer_workers:
                if ep not in contain_eps:
                    ep_transmit_times[ep].append(input_sizes[key] * 1.0 / self._avg_net_speed)

        min_data_count = options.scheduler.steal_min_data_count
        threshold_ratio = options.scheduler.stealable_threshold
        ep_total_time = dict()
        for ep in self._worker_metrics:
            calc_stats = self._worker_metrics[ep].get('stats', {}).get('calc_speed.' + op_name, {})
            if calc_stats['count'] < min_data_count:
                calc_speed = self._avg_calc_speeds[op_name]
            else:
                calc_speed = calc_stats['mean']
            transfer_time = sum(ep_transmit_times[ep] or ()) + output_size / self._avg_net_speed
            calc_time = output_size / calc_speed
            if calc_time < threshold_ratio * transfer_time:
                return None
            ep_total_time[ep] = sum(ep_transmit_times[ep] or ()) + output_size / self._avg_net_speed \
                + output_size / calc_speed
        sort_items = sorted([(t, ep) for ep, t in ep_total_time.items()])
        return [item[-1] for item in sort_items if item[0] == sort_items[0][0]] if sort_items else []
