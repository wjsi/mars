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

from ..graph import DAG
from ..operands import VirtualOperand

logger = logging.getLogger(__name__)


class DataTracker:
    def __init__(self):
        self._graph = DAG()
        self._key_to_chunks = dict()
        self._chunk_key_op_refs = dict()
        self._target_chunk_keys = set()

    def __contains__(self, item):
        return item in self._chunk_key_op_refs

    @property
    def tracked_keys(self):
        return set(self._chunk_key_op_refs.keys())

    def update_graph(self, new_graph, new_target_chunk_keys):
        ref_chunks = set()
        for chunk in new_graph:
            if chunk.key in self._chunk_key_op_refs:
                ref_chunks.add(chunk)
            elif chunk.key in self._target_chunk_keys and chunk.key not in new_target_chunk_keys:
                ref_chunks.add(chunk)
                self._chunk_key_op_refs[chunk.key] = set()

        for chunk in ref_chunks:
            for succ in new_graph.iter_successors(chunk):
                self._chunk_key_op_refs[chunk.key].add(succ.op.key)

        for key in new_target_chunk_keys:
            self._chunk_key_op_refs.pop(key, None)

        self._graph = new_graph
        self._target_chunk_keys = set(new_target_chunk_keys)

        key_to_chunks = dict()
        for c in self._graph:
            try:
                chunks_set = key_to_chunks[c.key]
            except KeyError:
                chunks_set = key_to_chunks[c.key] = set()
            chunks_set.add(c)
        self._key_to_chunks = key_to_chunks

    def add_tracks(self, chunks):
        ref_counts = dict()
        chunk_keys = set(c.key for c in chunks)

        for key in chunk_keys:
            if key in self._target_chunk_keys:
                continue
            try:
                track_op_keys = set(succ.op.key for c in self._key_to_chunks[key]
                                    for succ in self._graph.iter_successors(c))
            except KeyError:
                raise
            if key not in self._chunk_key_op_refs:
                self._chunk_key_op_refs[key] = track_op_keys
            else:
                self._chunk_key_op_refs[key] |= track_op_keys
            ref_counts[key] = len(self._chunk_key_op_refs[key])

        for key in chunk_keys:
            if not self._chunk_key_op_refs.get(key):
                self._chunk_key_op_refs.pop(key, None)

        return ref_counts

    def decref(self, ops):
        data_to_remove = set()
        for op in ops:
            for c in (op.inputs or ()):
                if c.key not in self._chunk_key_op_refs or c not in self._graph:
                    continue
                if isinstance(c.op, VirtualOperand):
                    continue
                ref_set = self._chunk_key_op_refs[c.key]
                ref_set.difference_update((op.key,))
                if not ref_set:
                    self._chunk_key_op_refs.pop(c.key, None)
                    data_to_remove.add(c.key)
        return data_to_remove
