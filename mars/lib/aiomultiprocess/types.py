# Copyright 2018 John Reese
# Licensed under the MIT license

import multiprocessing
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")
R = TypeVar("R")

Context = multiprocessing.context.BaseContext
Queue = multiprocessing.Queue

TaskID = NewType("TaskID", int)
QueueID = NewType("QueueID", int)

TracebackStr = str

PoolTask = Optional[Tuple[TaskID, Callable[..., R], Sequence[T], Dict[str, T]]]
PoolResult = Tuple[TaskID, Optional[R], Optional[TracebackStr]]


Unit = NamedTuple('Unit', [
    ('target', Callable),
    ('args', Sequence[Any]),
    ('kwargs', Dict[str, Any]),
    ('namespace', Any),
    ('initializer', Optional[Callable]),
    ('initargs', Sequence[Any]),
    ('runner', Optional[Callable]),
])


class ProxyException(Exception):
    pass
