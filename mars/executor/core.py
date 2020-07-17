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

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from ..utils import classproperty

try:
    import gevent
except ImportError:  # pragma: no cover
    gevent = None

if gevent:
    from ..actors.pool.gevent_pool import GeventThreadPool


class OperandState(Enum):
    __order__ = 'UNSCHEDULED READY RUNNING FINISHED CACHED FREED FATAL CANCELLING CANCELLED'

    UNSCHEDULED = 'unscheduled'
    READY = 'ready'
    RUNNING = 'running'
    FINISHED = 'finished'
    CACHED = 'cached'
    FREED = 'freed'
    FATAL = 'fatal'
    CANCELLING = 'cancelling'
    CANCELLED = 'cancelled'

    @classproperty
    def STORED_STATES(self):
        """
        States on which the data of the operand is stored
        """
        return self.FINISHED, self.CACHED

    @classproperty
    def SUCCESSFUL_STATES(self):
        """
        States on which the operand is executed successfully
        """
        return self.FINISHED, self.CACHED, self.FREED

    @classproperty
    def TERMINATED_STATES(self):
        """
        States on which the operand has already terminated
        """
        return self.FINISHED, self.CACHED, self.FREED, self.FATAL, self.CANCELLED


class SyncProviderType(Enum):
    THREAD = 0
    GEVENT = 1
    MOCK = 2


class ExecutorSyncProvider(object):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        raise NotImplementedError

    @classmethod
    def semaphore(cls, value):
        raise NotImplementedError

    @classmethod
    def lock(cls):
        raise NotImplementedError

    @classmethod
    def rlock(cls):
        raise NotImplementedError

    @classmethod
    def event(cls):
        raise NotImplementedError

    @classmethod
    def queue(cls, *args, **kwargs):
        raise NotImplementedError


class EventQueue(list):
    def __init__(self, event_cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if event_cls is not None:
            self._has_value = event_cls()
            if len(self) > 0:
                self._has_value.set()
        else:
            self._has_value = None

    def append(self, item):
        super().append(item)
        if self._has_value is not None:
            self._has_value.set()

    def insert(self, index: int, item) -> None:
        super().insert(index, item)
        if self._has_value is not None:
            self._has_value.set()

    def pop(self, index=-1):
        item = super().pop(index)
        if self._has_value is not None and len(self) == 0:
            self._has_value.clear()
        return item

    def clear(self) -> None:
        super().clear()
        if self._has_value is not None:
            self._has_value.clear()

    def wait(self, timeout=None):
        if self._has_value is not None:
            self._has_value.wait(timeout)

    def errored(self):
        if self._has_value is not None:
            self._has_value.set()


class ThreadExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return ThreadPoolExecutor(n_workers)

    @classmethod
    def semaphore(cls, value):
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        return threading.Lock()

    @classmethod
    def rlock(cls):
        return threading.RLock()

    @classmethod
    def event(cls):
        return threading.Event()

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(threading.Event, *args, **kwargs)


class GeventExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return GeventThreadPool(n_workers)

    @classmethod
    def semaphore(cls, value):
        # as gevent threadpool is the **real** thread, so use threading.Semaphore
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        # as gevent threadpool is the **real** thread, so use threading.Lock
        return threading.Lock()

    @classmethod
    def rlock(cls):
        # as gevent threadpool is the **real** thread, so use threading.RLock
        return threading.RLock()

    @classmethod
    def event(cls):
        # as gevent threadpool is the **real** thread, so use threading.Event
        import gevent.event
        return gevent.event.Event()

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(threading.Event, *args, **kwargs)


class MockThreadPoolExecutor(object):
    class _MockResult(object):
        def __init__(self, result=None, exc_info=None):
            self._result = result
            self._exc_info = exc_info

        def result(self, *_):
            if self._exc_info is not None:
                raise self._exc_info[1] from None
            else:
                return self._result

        def exception_info(self, *_):
            return self._exc_info

        def add_done_callback(self, callback):
            callback(self)

    def __init__(self, *_):
        pass

    def submit(self, fn, *args, **kwargs):
        try:
            return self._MockResult(fn(*args, **kwargs))
        except:  # noqa: E722
            return self._MockResult(None, sys.exc_info())

    @classmethod
    def queue(cls, *args, **kwargs):
        return EventQueue(None, *args, **kwargs)


class MockExecutorSyncProvider(ThreadExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return MockThreadPoolExecutor(n_workers)


_sync_provider = {
    SyncProviderType.MOCK: MockExecutorSyncProvider,
    SyncProviderType.THREAD: ThreadExecutorSyncProvider,
    SyncProviderType.GEVENT: GeventExecutorSyncProvider,
}


def get_sync_provider(provider_type):
    return _sync_provider[provider_type]
