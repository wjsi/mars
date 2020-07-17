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

import contextlib

from ...actors import ActorNotExist
from ...errors import WorkerDead


@contextlib.contextmanager
def rewrite_worker_errors(ignore_error=False):
    rewrite = False
    try:
        yield
    except (BrokenPipeError, ConnectionRefusedError, ActorNotExist, TimeoutError):
        # we don't raise here, as we do not want
        # the actual stack be dumped
        rewrite = not ignore_error
    if rewrite:
        raise WorkerDead


_op_cls_to_actor = dict()


def get_operand_actor_class(op_cls):
    try:
        return _op_cls_to_actor[op_cls]
    except KeyError:
        for super_cls in op_cls.__mro__:
            try:
                actor_cls = _op_cls_to_actor[op_cls] = _op_cls_to_actor[super_cls]
                return actor_cls
            except KeyError:
                continue
        raise KeyError(f'Operand type {op_cls} not supported')  # pragma: no cover


def register_operand_class(op_cls, actor_cls):
    _op_cls_to_actor[op_cls] = actor_cls
