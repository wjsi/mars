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
import typing

from ..graph import DAG
from ..operands import VirtualOperand, ShuffleProxy

logger = logging.getLogger(__name__)


class TrackNode:
    __slots__ = 'ref_keys', 'is_target'

    def __init__(self, ref_keys=None, is_target=False):
        self.ref_keys = set(ref_keys or ())
        self.is_target = is_target


class DataTracker:
    _key_to_node: typing.Dict[str, TrackNode]

    def __init__(self):
        self._key_to_node = dict()
        self._target_keys = set()

    def __contains__(self, item):
        return item in self._key_to_node

    @property
    def tracked_keys(self):
        return set(self._key_to_node.keys())

    def update_graph(self, new_graph: DAG, new_target_chunk_keys: set):
        self._target_keys = new_target_chunk_keys
        for chunk in new_graph:
            try:
                node = self._key_to_node[chunk.key]
            except KeyError:
                continue
            node.ref_keys = \
                set(succ.key for succ in new_graph.iter_successors(chunk))
            node.is_target = chunk.key in self._target_keys

    def add_track(self, graph: DAG, chunk):
        keys_to_delete = []

        if isinstance(chunk.op, VirtualOperand):
            return []

        successors = graph.successors(chunk)
        if len(successors) == 1 and isinstance(successors[0].op, ShuffleProxy):
            return []

        if chunk.key not in self._target_keys and len(successors) == 0:
            keys_to_delete.append(chunk.key)

        self._key_to_node[chunk.key] = TrackNode(
            [succ.key for succ in successors], chunk.key in self._target_keys,
        )
        for pred_chunk in graph.iter_predecessors(chunk):
            try:
                pred_node = self._key_to_node[pred_chunk.key]
            except KeyError:
                continue

            if pred_node.is_target:
                continue
            if all(succ_key in self._key_to_node for succ_key in pred_node.ref_keys):
                del self._key_to_node[pred_chunk.key]
                keys_to_delete.append(pred_chunk.key)

        return keys_to_delete

    def remove_tracks(self, chunk_keys):
        logger.warning('UNTRACK %s', chunk_keys)
        for key in chunk_keys:
            self._key_to_node.pop(key, None)
        self._target_keys.difference_update(chunk_keys)
