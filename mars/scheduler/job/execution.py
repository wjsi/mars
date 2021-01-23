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

from ...graph import DAG
from ...operands import Fetch, ShuffleProxy, SuccessorsExclusive
from ...utils import get_chunk_shuffle_key, build_fetch_chunk, serialize_graph
from ..analyzer import GraphAnalyzer
from ..assigner import AssignerActor
from ..chunklifecycle import ChunkLifecycleActor
from ..resource import ResourceActor

logger = logging.getLogger(__name__)


class JobExecution:
    def __init__(self, session_id, job_id, chunk_graph, initials=None, actor_ctx=None,
                 job_address=None):
        self._actor_ctx = actor_ctx
        self._session_id = session_id
        self._job_id = job_id
        self._job_address = job_address
        self._chunk_graph = chunk_graph

        self._assigner_ref = actor_ctx.actor_ref(AssignerActor.gen_uid(session_id))
        self._chunk_lifecycle_ref = actor_ctx.actor_ref(ChunkLifecycleActor.gen_uid(self._session_id))
        self._resource_ref = actor_ctx.actor_ref(ResourceActor.default_uid())

        self._op_key_to_chunks = dict()
        for c in chunk_graph:
            try:
                op_key_list = self._op_key_to_chunks[c.op.key]
            except KeyError:
                op_key_list = self._op_key_to_chunks[c.op.key] = []
            op_key_list.append(c)
        self._operand_infos = dict()

        self._chunk_lifecycle_ref.incref([c.key for c in chunk_graph])
        self._ready_op_keys = set(c.op.key for c in (initials or chunk_graph.iter_indep()))
        self._stored_chunk_keys = set()
        self._released_chunk_keys = set()

        self._graph_analyze_pool = actor_ctx.threadpool(1)
        self._graph_analyze_pool.submit(self._analyze_graph).result()

    def _get_worker_slots(self):
        metrics = self._resource_ref.get_workers_meta()
        return dict((ep, int(metrics[ep]['hardware']['cpu_total'])) for ep in metrics)

    def _assign_initial_workers(self, analyzer):
        # collect external inputs for eager mode
        operand_infos = self._operand_infos
        chunk_graph = self._chunk_graph

        initial_chunks = [c for c in chunk_graph
                          if chunk_graph.count_predecessors(c) == 0]

        if all(c.op.expect_worker is not None for c in initial_chunks):
            assignments = {c.op.key: c.op.expect_worker for c in initial_chunks}
        else:
            assignments = analyzer.calc_operand_assignments(self._ready_op_keys)
        for idx, (k, v) in enumerate(assignments.items()):
            operand_infos[k]['optimize']['placement_order'] = idx
            operand_infos[k]['target_worker'] = v
        return assignments

    def _analyze_graph(self, **kwargs):
        operand_infos = self._operand_infos
        chunk_graph = self._chunk_graph

        # remove fetch chunk if exists
        if any(isinstance(c.op, Fetch) for c in chunk_graph):
            chunk_graph = chunk_graph.copy()
            for c in list(chunk_graph):
                if isinstance(c.op, Fetch):
                    chunk_graph.remove_node(c)

        if len(chunk_graph) == 0:
            return

        for c in chunk_graph:
            k = c.op.key
            succ_size = chunk_graph.count_successors(c)
            if k not in operand_infos:
                operand_infos[k] = dict(
                    optimize=dict(
                        depth=0, demand_depths=(), successor_size=succ_size, descendant_size=0,
                    ),
                    calc_device='cuda' if c.op.gpu else 'cpu',
                    op_name=type(c.op).__name__,
                )
            else:
                operand_infos[k]['optimize']['successor_size'] = succ_size

        worker_slots = self._get_worker_slots()
        if not worker_slots:
            raise RuntimeError('No worker attached for execution')

        self._assigned_workers = set(worker_slots)
        analyzer = GraphAnalyzer(chunk_graph, worker_slots)

        for k, v in analyzer.calc_depths().items():
            operand_infos[k]['optimize']['depth'] = v

        for k, v in analyzer.calc_descendant_sizes().items():
            operand_infos[k]['optimize']['descendant_size'] = v

        if kwargs.get('do_placement', True):
            logger.debug('Placing initial chunks for job %s', self._job_id)
            self._assign_initial_workers(analyzer)

    def _collect_operand_io_meta(self, chunks):
        # collect operand i/o information
        predecessor_keys = set()
        successor_keys = set()
        input_chunk_keys = set()
        shared_input_chunk_keys = set()
        pure_dep_chunk_keys = set()
        chunk_keys = set()
        shuffle_keys = dict()
        predecessors_to_successors = dict()

        graph = self._chunk_graph

        for c in chunks:
            # handling predecessor args
            for pn in graph.iter_predecessors(c):
                if not isinstance(pn.op, Fetch):
                    predecessor_keys.add(pn.op.key)
                input_chunk_keys.add(pn.key)
                if graph.count_successors(pn) > 1:
                    shared_input_chunk_keys.add(pn.key)

            for inp, is_dep in zip(c.op.inputs or (), c.op.pure_depends):
                if is_dep and inp.key in input_chunk_keys:
                    pure_dep_chunk_keys.add(inp.key)

            # handling successor args
            for sn in graph.iter_successors(c):
                successor_keys.add(sn.op.key)
            if isinstance(c.op, ShuffleProxy):
                for sn in graph.iter_successors(c):
                    shuffle_keys[sn.op.key] = get_chunk_shuffle_key(sn)
            if isinstance(c.op, SuccessorsExclusive):
                for sn in graph.iter_successors(c):
                    predecessors_to_successors[sn.inputs[0].op.key] = sn.op.key

            chunk_keys.update(co.key for co in c.op.outputs)

        io_meta = dict(
            predecessors=list(predecessor_keys),
            successors=list(successor_keys),
            input_chunks=list(input_chunk_keys),
            pure_dep_chunk_keys=list(pure_dep_chunk_keys),
            shared_input_chunks=list(shared_input_chunk_keys),
            chunks=list(chunk_keys),
        )
        if shuffle_keys:
            io_meta['shuffle_keys'] = [shuffle_keys.get(k) for k in io_meta['successors']]
        if predecessors_to_successors:
            io_meta['predecessors_to_successors'] = predecessors_to_successors
        return io_meta

    def _submit_operands(self, op_keys):
        operand_infos = self._operand_infos

        for op_key in op_keys:
            if 'io_meta' in operand_infos[op_key]:
                continue
            operand_infos[op_key]['io_meta'] = \
                self._collect_operand_io_meta(self._op_key_to_chunks[op_key])

        res_applications = [(key, operand_infos[key]) for key in self._ready_op_keys]
        self._assigner_ref.apply_for_multiple_resources(
            self._session_id, self._job_id, res_applications, _tell=True)

    def submit_all(self):
        self._submit_operands(self._ready_op_keys)

    def get_executable_operand_dag(self, op_key, input_chunk_keys=None, serialize=True):
        """
        Make an operand into a worker-executable dag
        :param op_key: operand key
        :param input_chunk_keys: actual input chunks, None if use all chunks in input
        :param serialize: whether to return serialized dag
        """
        graph = DAG()
        input_mapping = dict()
        output_keys = set()

        input_chunk_keys = set(input_chunk_keys) if input_chunk_keys is not None else None
        for c in self._op_key_to_chunks[op_key]:
            inputs = []
            for inp in set(c.op.inputs or ()):
                try:
                    inp_chunk = input_mapping[(inp.key, inp.id)]
                except KeyError:
                    inp_chunk = input_mapping[(inp.key, inp.id)] \
                        = build_fetch_chunk(inp, input_chunk_keys).data
                    graph.add_node(inp_chunk)
                inputs.append(inp_chunk)

            for out in set(c.op.outputs or ()):
                if (out.key, out.id) not in output_keys:
                    output_keys.add((out.key, out.id))
                    graph.add_node(out)
                    for inp in inputs:
                        graph.add_edge(inp, out)
        if serialize:
            return self._graph_analyze_pool.submit(serialize_graph, graph).result()
        else:
            return graph

    def submit_operand_to_worker(self, op_key, worker, input_metas):
        from ...worker.execution import ExecutionActor
        # submit job
        exec_graph = self.get_executable_operand_dag(op_key)
        execution_ref = self._actor_ctx.actor_ref(ExecutionActor.default_uid(), address=worker)
        io_meta = self._operand_infos[op_key]['io_meta']

        execution_ref.execute_graph(
            self._session_id, self._job_id, op_key, exec_graph, io_meta, input_metas,
            calc_device=self._operand_infos[op_key]['calc_device'], _tell=True
        )

        self._ready_op_keys.difference_update([op_key])

    def set_operand_finished(self, op_key):
        keys_to_release = set()
        # record chunk keys as created
        for c in self._op_key_to_chunks[op_key]:
            if self._chunk_graph.count_successors(c) == 0:
                keys_to_release.add(c.key)
            else:
                self._stored_chunk_keys.add(c.key)
        # check if any successors can be activated
        for c in self._op_key_to_chunks[op_key]:
            for succ in self._chunk_graph.iter_successors(c):
                if succ.op.key in self._ready_op_keys:
                    continue
                if all(inp.key in self._stored_chunk_keys for inp in c.inputs):
                    self._ready_op_keys.add(succ.op.key)
                    self._submit_operands([succ.op.key])

        # check if we can decref previous chunks
        for c in self._op_key_to_chunks[op_key]:
            for pred in self._chunk_graph.iter_predecessors(c):
                if all(out.key in self._stored_chunk_keys or out.key in self._released_chunk_keys
                       for out in self._chunk_graph.iter_successors(pred)):
                    keys_to_release.add(pred.key)

        self._chunk_lifecycle_ref.decref(keys_to_release)
        self._released_chunk_keys.update(keys_to_release)

    def clean_all(self):
        keys_to_remove = set()
        for c in self._chunk_graph:
            if c.key not in self._stored_chunk_keys:
                keys_to_remove.add(c.key)
