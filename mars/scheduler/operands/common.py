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

import base64
import logging
import time

from ...config import options
from ...errors import ExecutionInterrupted, DependencyMissing, WorkerDead
from ...operands import Operand
from ...utils import log_unhandled
from ..utils import GraphState, array_to_bytes, rewrite_worker_errors
from .base import BaseOperandActor
from .core import OperandState, OperandPosition, register_operand_class

logger = logging.getLogger(__name__)


class OperandActor(BaseOperandActor):
    """
    Actor handling the whole lifecycle of a particular operand instance
    """

    def __init__(self, session_id, graph_id, op_key, op_info, worker=None):
        super(OperandActor, self).__init__(session_id, graph_id, op_key, op_info, worker=worker)
        io_meta = self._io_meta
        self._input_chunks = io_meta['input_chunks']
        self._chunks = io_meta['chunks']

        # worker the operand expected to be executed on
        self._target_worker = op_info.get('target_worker')
        self._retries = op_info['retries']
        self._assigned_workers = set()

        # ref of ExecutionActor on worker
        self._execution_ref = None

        self._data_sizes = None

        self._input_worker_scores = dict()
        self._worker_scores = dict()

    @property
    def retries(self):
        return self._retries

    @retries.setter
    def retries(self, value):
        futures = []
        self._retries = value
        self._info['retries'] = value

        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write(
                '%s/retries' % self._op_path, str(value), _tell=True, _wait=False))

        retry_timestamp = time.time()
        self._info['retry_timestamp'] = retry_timestamp
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/retry_timestamp' % self._op_path, str(value),
                                                    _tell=True, _wait=False))
        [f.result() for f in futures]

    def append_graph(self, graph_key, op_info):
        from ..graph import GraphActor

        if self._position != OperandPosition.TERMINAL:
            self._position = op_info.get('position')
        graph_ref = self.get_actor_ref(GraphActor.gen_uid(self._session_id, graph_key))
        self._graph_refs.append(graph_ref)
        self._pred_keys.update(op_info['io_meta']['predecessors'])
        self._succ_keys.update(op_info['io_meta']['successors'])
        if self._state not in OperandState.STORED_STATES and self._state != OperandState.RUNNING:
            self._state = op_info['state']

    def start_operand(self, state=None, **kwargs):
        target_worker = kwargs.get('target_worker')
        if target_worker:
            self._target_worker = target_worker
        return super(OperandActor, self).start_operand(state=state, **kwargs)

    def add_finished_predecessor(self, op_key, worker, output_sizes=None):
        super(OperandActor, self).add_finished_predecessor(op_key, worker, output_sizes=output_sizes)
        if all(k in self._finish_preds for k in self._pred_keys):
            if self.state != OperandState.UNSCHEDULED:
                return True
            # all predecessors done, the operand can be executed now
            self.start_operand(OperandState.READY)
            return True
        self.update_demand_depths(self._info.get('optimize', {}).get('depth', 0))
        return False

    def add_finished_successor(self, op_key, worker):
        super(OperandActor, self).add_finished_successor(op_key, worker)
        if self._position != OperandPosition.TERMINAL and \
                all(k in self._finish_succs for k in self._succ_keys):
            # make sure that all prior states are terminated (in case of failover)
            states = []
            for graph_ref in self._graph_refs:
                states.extend(graph_ref.get_operand_states(self._succ_keys))
            # non-terminal operand with all successors done, the data can be freed
            if all(k in OperandState.TERMINATED_STATES for k in states) and self._is_worker_alive():
                self.ref().free_data(_tell=True)

    def _get_priority_data(self):
        try:
            priority_data = self._info['optimize']
        except KeyError:
            priority_data = dict()

        return tuple([
            priority_data.get('depth', 0),
            priority_data.get('demand_depths', ()),
            -priority_data.get('successor_size', 0),
            -priority_data.get('placement_order', 0),
            priority_data.get('descendant_size', 0),
        ])

    def update_demand_depths(self, depth):
        """
        Update the depth of operand demanding data, or demanding the descendant of data
        produced by the current operand
        :param depth: depth to update
        """
        demand_depths = list(self._info.get('optimize', {}).get('demand_depths', ()))
        if not demand_depths:
            demand_depths = [depth]
        else:
            idx = 0
            for idx, v in enumerate(demand_depths):
                if v <= depth:
                    break
            if demand_depths[idx] == depth:
                return
            elif demand_depths[idx] > depth:
                demand_depths.append(depth)
            else:
                demand_depths.insert(idx, depth)
        try:
            optimize_data = self._info['optimize']
        except KeyError:
            optimize_data = self._info['optimize'] = dict()

        optimize_data['demand_depths'] = tuple(demand_depths)
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '%s/optimize/demand_depths' % self._op_path,
                base64.b64encode(array_to_bytes('I', demand_depths)), _tell=True, _wait=False)

        # if the operand is already submitted to TaskPoolActor, we need to update the priority
        self._assigner_ref.update_operand_priority(
            self._session_id, self._op_key, self._get_priority_data(), _tell=True)

        futures = []
        if self.state != OperandState.READY:
            # send update command to predecessors
            for in_key in self._pred_keys:
                futures.append(self._get_operand_actor(in_key).update_demand_depths(
                    depth, _tell=True, _wait=False))
            [f.result() for f in futures]

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        """
        Calculate likelihood of the operand being sent to workers
        :param input_key: input key that carries the scores
        :param worker_scores: score of every worker on input key
        :param depth: maximal propagate depth
        """
        if self.worker:
            # worker already assigned, there should be no other possibilities
            self._worker_scores = {self.worker: 1.0}
        elif self._target_worker:
            # worker already proposed, there should be no other possibilities
            self._worker_scores = {self._target_worker: 1.0}
        else:
            # aggregate the score from input to the score of current operand
            old_scores = self._input_worker_scores.get(input_key, {})
            self._input_worker_scores[input_key] = worker_scores
            all_keys = set(old_scores.keys()) | set(worker_scores)
            for k in all_keys:
                delta = (worker_scores.get(k, 0) - old_scores.get(k, 0)) * 1.0 / len(self._pred_keys)
                self._worker_scores[k] = self._worker_scores.get(k, 0) + delta
                if self._worker_scores[k] < 1e-6:
                    del self._worker_scores[k]

        live_workers = set(self._assigner_ref.filter_alive_workers(list(self._worker_scores.keys())))
        self._worker_scores = dict((k, v) for k, v in self._worker_scores.items()
                                   if k in live_workers)

        if self._worker_scores and depth:
            # push down to successors
            futures = []
            for succ_key in self._succ_keys:
                futures.append(self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, self._worker_scores, depth=depth - 1, _tell=True, _wait=False))
            [f.result() for f in futures]
        # pick the worker with largest likelihood
        max_score = 0
        max_worker = None
        for k, v in self._worker_scores.items():
            if v > max_score:
                max_score = v
                max_worker = k
        if max_score > 0.5:
            logger.debug('Operand %s(%s) now owning a dominant worker %s. scores=%r',
                         self._op_key, self._op_name, max_worker, self._worker_scores)
            return self._input_chunks, max_worker

    def _is_worker_alive(self):
        return bool(self._assigner_ref.filter_alive_workers([self.worker], refresh=True))

    def move_failover_state(self, from_states, state, new_target, dead_workers):
        """
        Move the operand into new state when executing fail-over step
        :param from_states: the source states the operand should be in, when not match, we stopped.
        :param state: the target state to move
        :param new_target: new target worker proposed for worker
        :param dead_workers: list of dead workers
        :return:
        """
        dead_workers = set(dead_workers)
        if self.state not in from_states:
            logger.debug('From state not matching (%s not in %r), operand %s skips failover step',
                         self.state.name, [s.name for s in from_states], self._op_key)
            return
        if self.state in (OperandState.RUNNING, OperandState.FINISHED):
            if state != OperandState.UNSCHEDULED and self.worker not in dead_workers:
                logger.debug('Worker %s of operand %s still alive, skip failover step',
                             self.worker, self._op_key)
                return
            elif state == OperandState.RUNNING:
                # move running operand in dead worker to ready
                state = OperandState.READY

        if new_target and self._target_worker != new_target:
            logger.debug('Target worker of %s reassigned to %s', self._op_key, new_target)
            self._target_worker = new_target
            self._info['target_worker'] = new_target

            for succ_key in self._succ_keys:
                self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, {new_target: 1.0}, _wait=False)

            target_updated = True
        else:
            target_updated = False

        if self.state == state == OperandState.READY:
            if not self._target_worker:
                if self._assigned_workers - dead_workers:
                    logger.debug('Operand %s still have alive workers assigned %r, skip failover step',
                                 self._op_key, list(self._assigned_workers - dead_workers))
                    return
            else:
                if not target_updated and self._target_worker not in dead_workers:
                    logger.debug('Target of operand %s (%s) not dead, skip failover step',
                                 self._op_key, self._target_worker)
                    return

        if dead_workers:
            futures = []
            # remove executed traces in neighbor operands
            for out_key in self._succ_keys:
                futures.append(self._get_operand_actor(out_key).remove_finished_predecessor(
                    self._op_key, _tell=True, _wait=False))
            for in_key in self._pred_keys:
                futures.append(self._get_operand_actor(in_key).remove_finished_successor(
                    self._op_key, _tell=True, _wait=False))
            if self._position == OperandPosition.TERMINAL:
                for graph_ref in self._graph_refs:
                    futures.append(graph_ref.remove_finished_terminal(
                        self._op_key, _tell=True, _wait=False))
            [f.result() for f in futures]

        # actual start the new state
        self.start_operand(state)

    def free_data(self, state=OperandState.FREED):
        """
        Free output data of current operand
        :param state: target state
        """
        if self.state == OperandState.FREED:
            return
        if state == OperandState.CANCELLED:
            can_be_freed = True
        else:
            can_be_freed_states = [graph_ref.check_operand_can_be_freed(self._succ_keys) for
                                   graph_ref in self._graph_refs]
            if None in can_be_freed_states:
                can_be_freed = None
            else:
                can_be_freed = all(can_be_freed_states)
        if can_be_freed is None:
            self.ref().free_data(state, _delay=1, _tell=True)
            return
        elif not can_be_freed:
            return

        self.start_operand(state)

        stored_keys = self._io_meta.get('data_targets')
        if stored_keys:
            self._free_data_in_worker(stored_keys)

    def _get_execution_ref(self, uid=None, address=None):
        """
        Get ref of ExecutionActor on assigned worker
        """
        if address is None and self._execution_ref is not None:
            return self._execution_ref
        ref = self.promise_ref(self._get_raw_execution_ref(uid, address=address or self.worker))
        if address is None:
            self._execution_ref = ref
        return ref

    def _get_target_predicts(self, worker):
        target_predicts = dict()
        if options.scheduler.enable_active_push:
            # if active push enabled, we calculate the most possible target
            futures = []
            for succ_key in self._succ_keys:
                futures.append(self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, {worker: 1.0}, _wait=False))
            for succ_key, future in zip(self._succ_keys, futures):
                succ_worker_predict = future.result()
                if not succ_worker_predict:
                    continue
                keys, target = succ_worker_predict
                if target == worker:
                    continue
                for k in keys:
                    if k not in self._chunks:
                        continue
                    if k not in target_predicts:
                        target_predicts[k] = set()
                    target_predicts[k].add(target)
        if not target_predicts:
            target_predicts = None
        else:
            logger.debug('Receive active pushing list for operand %s: %r',
                         self._op_key, target_predicts)
        return target_predicts

    @log_unhandled
    def handle_worker_accept(self, worker):
        if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
            self.ref().start_operand(_tell=True)
            return

        if worker != self.worker:
            self._execution_ref = None
        self.worker = worker
        self._assigned_workers = set()

        target_predicts = self._get_target_predicts(worker)

        # prepare meta broadcasts
        broadcast_eps = set()
        for succ_key in self._succ_keys:
            broadcast_eps.add(self.get_scheduler(self.gen_uid(self._session_id, succ_key)))
        broadcast_eps.difference_update({self.address})
        broadcast_eps = tuple(broadcast_eps)

        chunk_keys, broadcast_ep_groups = [], []
        for chunk_key in self._chunks:
            chunk_keys.append(chunk_key)
            broadcast_ep_groups.append(broadcast_eps)
        broadcast_chunk_keys = [k for k in chunk_keys if not isinstance(k, tuple)]
        if broadcast_chunk_keys:
            self.chunk_meta.batch_set_chunk_broadcasts(
                self._session_id, broadcast_chunk_keys, broadcast_ep_groups,
                _tell=True, _wait=False)

        logger.debug('Start running operand %s on %s', self._op_key, worker)

        # register target worker
        self._execution_ref = self._get_execution_ref()
        try:
            with rewrite_worker_errors():
                if target_predicts:
                    self._get_execution_ref().send_data_to_workers(
                        self._session_id, self._op_key, target_predicts, _tell=True)
        except WorkerDead:
            self._resource_ref.detach_dead_workers([self.worker], _tell=True)
            return

        # here we start running immediately to avoid accidental state change
        # and potential submission
        self.start_operand(OperandState.RUNNING)

    @log_unhandled
    def _on_ready(self):
        self.worker = None
        self._execution_ref = None

        # if under retry, give application a delay
        delay = options.scheduler.retry_delay if self.retries else 0
        if delay > 0 or self._pred_keys:
            self.ref().submit_operand(_tell=True, _delay=delay)
        else:
            self._assigner_ref.submit_initial(
                self._op_key, self._get_priority_data(), self._target_worker, _tell=True)

    def submit_operand(self):
        op_io_meta = self._io_meta = self._info['io_meta']
        try:
            key_to_metas = op_io_meta['input_data_metas']
            input_chunk_keys = [k[0] if isinstance(k, tuple) else k for k in key_to_metas.keys()]
            exec_graph_future = None
        except KeyError:
            input_chunk_keys = input_data_keys = op_io_meta['input_chunks']
            exec_graph_future = self._graph_refs[0].get_executable_operand_dag(self._op_key, _wait=False)

            if input_data_keys:
                key_to_metas = dict(zip(
                    input_data_keys,
                    self.chunk_meta.batch_get_chunk_meta(self._session_id, input_data_keys)
                ))
            else:
                key_to_metas = dict()

            if any(meta is None for meta in key_to_metas.values()):
                logger.warning('DependencyMissing met, operand %s will be back to UNSCHEDULED.',
                               self._op_key)
                self._assigned_workers = set()
                self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
                return

        if exec_graph_future is None:
            exec_graph = self._graph_refs[0].get_executable_operand_dag(
                self._op_key, input_chunk_keys)
        else:
            exec_graph = exec_graph_future.result()

        self._assigned_workers = set(self._assigner_ref.submit_operand(
            self._op_key, exec_graph, self._info['io_meta'], self._get_priority_data(),
            key_to_metas, self._target_worker
        ))

    @log_unhandled
    def _on_running(self):
        self._execution_ref = self._get_execution_ref()

        @log_unhandled
        def _acceptor(data_sizes):
            if not self._is_worker_alive():
                return
            self._data_sizes = data_sizes
            self._io_meta['data_targets'] = list(data_sizes)
            self.start_operand(OperandState.FINISHED)

        @log_unhandled
        def _rejecter(*exc):
            # handling exception occurrence of operand execution
            exc_type = exc[0]
            if self.state == OperandState.CANCELLING:
                logger.warning('Execution of operand %s cancelled.', self._op_key)
                self.free_data(OperandState.CANCELLED)
                return

            if issubclass(exc_type, ExecutionInterrupted):
                # job cancelled: switch to cancelled
                logger.warning('Execution of operand %s interrupted.', self._op_key)
                self.free_data(OperandState.CANCELLED)
            elif issubclass(exc_type, DependencyMissing):
                logger.warning('Operand %s moved to UNSCHEDULED because of DependencyMissing.',
                               self._op_key)
                self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            else:
                logger.exception('Attempt %d: Unexpected error %s occurred in executing operand %s in %s',
                                 self.retries + 1, exc_type.__name__, self._op_key, self.worker, exc_info=exc)
                # increase retry times
                self.retries += 1
                if self.retries >= options.scheduler.retry_num:
                    # no further trial
                    self.state = OperandState.FATAL
                else:
                    self.state = OperandState.READY
                self.ref().start_operand(_tell=True)

        try:
            with rewrite_worker_errors():
                self._execution_ref.add_finish_callback(self._session_id, self._op_key, _promise=True) \
                    .then(_acceptor, _rejecter)
        except WorkerDead:
            logger.debug('Worker %s dead when adding callback for operand %s',
                         self.worker, self._op_key)
            self._resource_ref.detach_dead_workers([self.worker], _tell=True)
            self.start_operand(OperandState.READY)

    @log_unhandled
    def _on_finished(self):
        if self._last_state == OperandState.CANCELLING:
            self.start_operand(OperandState.CANCELLING)
            return

        futures = []
        # update pred & succ finish records to trigger further actions
        # record if successors can be executed
        for out_key in self._succ_keys:
            futures.append(self._get_operand_actor(out_key).add_finished_predecessor(
                self._op_key, self.worker, output_sizes=self._data_sizes,
                _tell=True, _wait=False))
        for in_key in self._pred_keys:
            futures.append(self._get_operand_actor(in_key).add_finished_successor(
                self._op_key, self.worker, _tell=True, _wait=False))
        # require more chunks to execute if the completion caused no successors to run
        if self._position == OperandPosition.TERMINAL:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.extend(self._add_finished_terminal())
        [f.result() for f in futures]

    @log_unhandled
    def _on_fatal(self):
        if self._last_state == OperandState.FATAL:
            return

        futures = []
        if self._position == OperandPosition.TERMINAL:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.extend(self._add_finished_terminal(final_state=GraphState.FAILED))
        # set successors to FATAL
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).stop_operand(
                OperandState.FATAL, _tell=True, _wait=False))
        [f.result() for f in futures]

    @log_unhandled
    def _on_cancelling(self):
        if self._last_state == OperandState.CANCELLING:
            return
        elif self._last_state == OperandState.CANCELLED:
            self.state = OperandState.CANCELLED
        elif self._last_state == OperandState.RUNNING:
            # send stop to worker
            self._execution_ref = self._get_execution_ref()
            logger.debug('Sending stop on operand %s to %s', self._op_key, self.worker)
            with rewrite_worker_errors(ignore_error=True):
                self._execution_ref.stop_execution(
                    self._session_id, self._op_key, _tell=True)
        elif self._last_state == OperandState.FINISHED:
            # delete data on cancelled
            self.ref().free_data(state=OperandState.CANCELLED, _tell=True)
        elif self._last_state == OperandState.READY:
            # stop application on workers
            self._assigner_ref.remove_task(self._session_id, self._op_key)
            self._assigned_workers = set()
            self.state = OperandState.CANCELLED
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)
        else:
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)

    @log_unhandled
    def _on_cancelled(self):
        futures = []
        if self._position == OperandPosition.TERMINAL:
            futures.extend(self._add_finished_terminal(final_state=GraphState.CANCELLED))
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).stop_operand(
                OperandState.CANCELLING, _tell=True, _wait=False))
        [f.result() for f in futures]

    def _on_unscheduled(self):
        self.worker = None

    def _add_finished_terminal(self, final_state=None):
        futures = []
        for graph_ref in self._graph_refs:
            futures.append(graph_ref.add_finished_terminal(
                self._op_key, final_state=final_state, _tell=True, _wait=False
            ))

        return futures


register_operand_class(Operand, OperandActor)
