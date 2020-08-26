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

# distutils: language = c++

from collections.abc import Iterable

from libc.stdint cimport uint64_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

cimport cython
import numpy as np
cimport numpy as np
import pyarrow as pa
from pyarrow.lib cimport CDataType, CArray, \
    pyarrow_unwrap_array, pyarrow_unwrap_data_type, pyarrow_wrap_array


cdef extern from 'arrow_utils_impl.hpp' namespace 'mars::dataframe::arrays' nogil:
    cdef enum SortType:
        QUICK_SORT = 0
        HEAP_SORT = 1
        MERGE_SORT = 2

    cdef enum BoundSide:
        LEFT = 0
        RIGHT = 1

    cdef shared_ptr[CArray] SortStringIndexes(vector[shared_ptr[CArray]] &data_chunks,
                                              shared_ptr[CDataType] &data_type_ptr,
                                              bint ascending,
                                              SortType sort_type)

    cdef shared_ptr[CArray] SearchStringIndexes(vector[shared_ptr[CArray]] &data_chunks,
                                                shared_ptr[CArray] &values,
                                                shared_ptr[CDataType] &data_type,
                                                vector[uint64_t] &sorter,
                                                BoundSide bound_side)


cdef uint64_t extract_chunked_array(array_obj, vector[shared_ptr[CArray]] &chunk_refs,
                                    shared_ptr[CDataType] &data_type_ptr):
    cdef uint64_t total_size = 0
    cdef shared_ptr[CArray] chunk_ref

    (&data_type_ptr)[0] = pyarrow_unwrap_data_type(array_obj.type)

    for chunk in array_obj.chunks:
        chunk_ref = pyarrow_unwrap_array(chunk)
        chunk_refs.push_back(chunk_ref)
        total_size += chunk_ref.get().length()
    return total_size


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
cpdef arrow_chunked_array_argsort(array_obj, bint ascending=True, kind=None):
    cdef uint64_t i
    cdef shared_ptr[CDataType] data_type_ptr
    cdef vector[shared_ptr[CArray]] chunk_refs
    cdef SortType sort_type

    kind = kind or 'quicksort'
    if kind == 'quicksort':
        sort_type = SortType.QUICK_SORT
    elif kind == 'heapsort':
        sort_type = SortType.HEAP_SORT
    elif kind == 'mergesort':
        sort_type = SortType.MERGE_SORT
    else:
        raise ValueError(f'Cannot accept kind {kind!r}')

    cdef uint64_t total_size = extract_chunked_array(array_obj, chunk_refs, data_type_ptr)
    if total_size == 0:
        return np.array([], dtype=np.int64)

    cdef shared_ptr[CArray] sort_result = \
        SortStringIndexes(chunk_refs, data_type_ptr, ascending, sort_type)

    if sort_result.get().length() == 0:
        data_type_id = data_type_ptr.get().id()
        data_type_str = bytes(data_type_ptr.get().ToString().c_str()).decode()
        raise NotImplementedError(f'argsort not implemented for data type {data_type_str} ({data_type_id})')

    return pyarrow_wrap_array(sort_result).to_numpy()


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
cpdef arrow_chunked_array_searchsorted(array_obj, value, side='left', sorter=None):
    cdef uint64_t i
    cdef BoundSide bound_side
    cdef shared_ptr[CDataType] data_type_ptr
    cdef vector[shared_ptr[CArray]] chunk_refs

    cdef np.ndarray sorter_np_array
    cdef vector[uint64_t] sorter_array

    value = value if isinstance(value, Iterable) and not isinstance(value, str) else [value]

    cdef object value_array = pa.array(value)

    if side == 'left':
        bound_side = BoundSide.LEFT
    elif side == 'right':
        bound_side = BoundSide.RIGHT
    else:
        raise ValueError(f'Cannot accept side {side!r}')

    extract_chunked_array(array_obj, chunk_refs, data_type_ptr)

    if sorter is not None:
        sorter_np_array = np.array(sorter)
        sorter_array.resize(len(sorter_np_array))
        for i in range(0, len(sorter_np_array)):
            sorter_array.push_back(sorter_np_array[i])

    cdef shared_ptr[CArray] value_array_ptr = pyarrow_unwrap_array(value_array)
    cdef shared_ptr[CArray] search_result = \
        SearchStringIndexes(chunk_refs, value_array_ptr, data_type_ptr, sorter_array, bound_side)

    if search_result.get().length() == 0:
        data_type_id = data_type_ptr.get().id()
        data_type_str = bytes(data_type_ptr.get().ToString().c_str()).decode()
        raise NotImplementedError(f'searchsorted not implemented for data type {data_type_str} ({data_type_id})')

    return pyarrow_wrap_array(search_result).to_numpy()
