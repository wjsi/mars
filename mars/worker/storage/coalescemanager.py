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

import os
import shutil
import time
from collections import defaultdict
from typing import Any, List, Dict, NamedTuple, Tuple, Union

from ...config import options
from ...serialize import dataserializer
from ...utils import mod_hash
from ..utils import WorkerActor


class CoalesceDataMeta(NamedTuple):
    data_key: Any
    file_path: Union[str, None]
    start: int
    end: int


class CoalesceFileInfo:
    __slots__ = 'file_path', 'file_size', 'block_metas', 'blocks'

    file_path: str
    file_size: int
    block_metas: List[CoalesceDataMeta]
    blocks: List

    def __init__(self, file_path=None, file_size=0, block_metas=None, blocks=None):
        self.file_path = file_path
        self.file_size = file_size
        self.block_metas = block_metas or []
        self.blocks = blocks or []


class ShuffleProxyProfile(NamedTuple):
    data_suffix_slot_map: Dict[Tuple, str]
    data_suffix_meta_map: Dict[Tuple, CoalesceDataMeta]
    slot_files_map: Dict[str, Dict[str, int]]
    data_prefixes: List[str]


class CoalesceManagerActor(WorkerActor):
    _proxy_key_profiles: Dict[Tuple, ShuffleProxyProfile]

    def __init__(self):
        super().__init__()

        self._data_prefix_to_proxy_key = dict()
        self._proxy_key_profiles = dict()

        self._combiner_refs = []

    def add_combiner_ref(self, ref):
        self._combiner_refs.append(self.ctx.actor_ref(ref))

    def register_key_slot_mapping(self, session_id, proxy_op_key, mapper_keys, shuffle_key_to_slot):
        self._proxy_key_profiles[(session_id, proxy_op_key)] = ShuffleProxyProfile(
            shuffle_key_to_slot,
            dict(),
            dict((slot, dict()) for slot in shuffle_key_to_slot.values()),
            mapper_keys
        )
        for mapper_key in mapper_keys:
            self._data_prefix_to_proxy_key[(session_id, mapper_key)] = proxy_op_key

    def register_new_file(self, session_id, proxy_key, slot_key, file_path,
                          metas: List[CoalesceDataMeta]):
        proxy_profile = self._proxy_key_profiles[(session_id, proxy_key)]
        data_meta_map = proxy_profile.data_suffix_meta_map
        for meta in metas:
            data_meta_map[meta.data_key[-1]] = CoalesceDataMeta(
                meta.data_key, file_path, meta.start, meta.end)

        proxy_profile.slot_files_map[slot_key][file_path] = len(metas)

    def get_combiner_ref_by_data_key(self, session_id, data_key):
        proxy_key = self._data_prefix_to_proxy_key[(session_id, data_key[0])]
        slot = self._proxy_key_profiles[(session_id, proxy_key)].data_suffix_slot_map[data_key[-1]]
        return self._combiner_refs[mod_hash(slot, len(self._combiner_refs))]

    def get_file_infos_from_keys(self, session_id, data_keys) -> List[CoalesceDataMeta]:
        metas = []
        for data_key in data_keys:
            proxy_key = self._data_prefix_to_proxy_key[(session_id, data_key[0])]
            metas.append(self._proxy_key_profiles[(session_id, proxy_key)]
                         .data_suffix_meta_map[data_key[-1]])
        return metas

    def delete_keys(self, session_id, data_keys):
        for data_key in data_keys:
            session_prefix_key = (session_id, data_key[0])
            proxy_key = self._data_prefix_to_proxy_key[session_prefix_key]

            proxy_profile = self._proxy_key_profiles[(session_id, proxy_key)]
            meta = proxy_profile.data_suffix_meta_map.pop(data_key[-1])
            slot = proxy_profile.data_suffix_slot_map[data_key[-1]]

            slot_files_map = proxy_profile.slot_files_map
            slot_files_map[slot][meta.file_path] -= 1
            if slot_files_map[slot][meta.file_path] == 0:
                os.unlink(meta.file_path)
                del slot_files_map[slot][meta.file_path]
            if not slot_files_map[slot]:
                del slot_files_map[slot]


class CoalesceCombinerActor(WorkerActor):
    _proxy_key_to_slot_cache_info: Dict[Tuple, Dict[str, CoalesceFileInfo]]

    def __init__(self, manager_ref):
        super().__init__()

        self._proxy_key_to_slot_cache_info = dict()
        self._compress = dataserializer.CompressType(options.worker.disk_compression)

        self._manager_ref = manager_ref

    def post_create(self):
        super().post_create()
        self._manager_ref = self.ctx.actor_ref(self._manager_ref)
        self._manager_ref.add_combiner_ref(self.ref(), _tell=True)

    def persist_block(self, session_id, proxy_key, slot_key, data_size, force=False):
        try:
            slot_cache_info = self._proxy_key_to_slot_cache_info[(session_id, proxy_key)]
        except KeyError:
            slot_cache_info = self._proxy_key_to_slot_cache_info[(session_id, proxy_key)] = dict()
        try:
            slot_info = slot_cache_info[slot_key]
        except KeyError:
            slot_info = slot_cache_info[slot_key] = CoalesceFileInfo()

        if not force and (
            slot_info.file_size + data_size <= options.worker.coalesce.max_file_size
            or len(slot_info.blocks) == 0
        ):
            return

        indices = list(range(len(slot_info.block_metas)))
        indices.sort(key=lambda ix: slot_info.block_metas[ix].data_key[-1])

        dirs = options.worker.spill_directory
        file_dir = dirs[mod_hash((session_id, proxy_key), len(dirs))]
        file_name = '@'.join([proxy_key, str(hash(slot_key)), str(int(time.time()))])
        temp_file_path = os.path.join(file_dir, 'writing', file_name)
        permanent_file_path = os.path.join(file_dir, file_name)

        with open(temp_file_path, 'wb') as file_obj:
            new_metas = []
            for idx in indices:
                block_meta = slot_info.block_metas[idx]

                block_meta.start = file_obj.tell()
                dataserializer.copy_to(file_obj, slot_info.blocks[idx], compress=self._compress)
                block_meta.end = file_obj.tell()

                new_metas.append(block_meta)

            slot_info.block_metas = new_metas
            slot_info.blocks = None

        shutil.move(temp_file_path, permanent_file_path)

        self._manager_ref.register_new_file(
            session_id, proxy_key, slot_key, permanent_file_path, new_metas)

        self._proxy_key_to_slot_cache_info[(session_id, proxy_key)][slot_key] = CoalesceFileInfo()

    def _put_data_in_cache(self, session_id, data_key, proxy_key, slot_key, data, data_size):
        self.persist_block(session_id, proxy_key, slot_key, data_size)
        slot_info = self._proxy_key_to_slot_cache_info[(session_id, proxy_key)][slot_key]
        slot_info.file_size += data_size
        slot_info.block_metas.append(CoalesceDataMeta(data_key, None, 0, 0))
        slot_info.blocks.append(data)

    def register_with_shared(self, session_id, data_key, proxy_key, slot_key):
        data_buf = self.shared_store.get_buffer(session_id, data_key)
        self._put_data_in_cache(session_id, data_key, proxy_key, slot_key, data_buf, data_buf.total_bytes)

    def register_with_bytes(self, session_id, data_key, proxy_key, slot_key, data):
        self._put_data_in_cache(session_id, data_key, proxy_key, slot_key, data, len(data))
