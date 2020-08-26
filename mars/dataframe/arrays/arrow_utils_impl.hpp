/* Copyright 1999-2020 Alibaba Group Holding Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _MARS_DATAFRAME_ARRAYS_ARROW_SORT_IMPL
#define _MARS_DATAFRAME_ARRAYS_ARROW_SORT_IMPL

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/type.h>

#define SMALL_MERGESORT 20


namespace mars {
namespace dataframe {
namespace arrays {

typedef std::pair<uint64_t, uint64_t> IndexPair;
typedef std::vector<IndexPair> IndexPairVector;
typedef std::vector<std::shared_ptr<arrow::Array>> ArrayChunks;

enum SortType {
    QUICK_SORT = 0,
    HEAP_SORT = 1,
    MERGE_SORT = 2
};

enum BoundSide {
    LEFT = 0,
    RIGHT = 1
};

template <bool ascending>
class StringArrayIndexComparer {
public:
    StringArrayIndexComparer(ArrayChunks &left_chunks, ArrayChunks &right_chunks)
        : _left_chunks(left_chunks), _right_chunks(right_chunks), _comparer(comparer_type()) {}

    bool operator() (const IndexPair &lhs, const IndexPair &rhs) {
        auto l_array = (arrow::StringArray *)_left_chunks[lhs.first].get();
        auto r_array = (arrow::StringArray *)_right_chunks[rhs.first].get();

        if (l_array->IsNull(lhs.second))
            return false;
        if (r_array->IsNull(rhs.second))
            return true;
        return _comparer(l_array->GetString(lhs.second), r_array->GetString(rhs.second));
    }
protected:
    typedef typename std::conditional<ascending,
                                      std::less<std::string>,
                                      std::greater<std::string>>::type comparer_type;

    ArrayChunks &_left_chunks, &_right_chunks;
    comparer_type _comparer;
};

template <typename Compare>
void MergeSortSubArray(IndexPairVector &array, IndexPairVector &buf,
    uint64_t left, uint64_t right, Compare &comparer) {

    if (right - left <= SMALL_MERGESORT) {
        // Use insertion sort for small arrays
        for (uint64_t i = left; i < right; i ++) {
            uint64_t min_pos = i;
            for (uint64_t j = i + 1; j < right; j ++) {
                if (comparer(array[j], array[min_pos])) {
                    min_pos = j;
                }
            }
            if (min_pos != i) {
                std::swap(array[i], array[min_pos]);
            }
        }
    }
    else {
        uint64_t mid = (left + right) / 2;
        MergeSortSubArray(array, buf, left, mid, comparer);
        MergeSortSubArray(array, buf, mid, right, comparer);
        std::merge(
            array.begin() + left, array.begin() + mid,
            array.begin() + mid, array.begin() + right,
            buf.begin() + left, comparer
        );
        std::copy(buf.begin() + left, buf.begin() + right, array.begin() + left);
    }
}

template <typename Compare>
void CallSortFunction(IndexPairVector &data_indexes, SortType sort_type, Compare comparer) {
    if (sort_type == SortType::QUICK_SORT) {
        std::sort(data_indexes.begin(), data_indexes.end(), comparer);
    }
    else if (sort_type == SortType::HEAP_SORT) {
        std::make_heap(data_indexes.begin(), data_indexes.end(), comparer);
        std::sort_heap(data_indexes.begin(), data_indexes.end(), comparer);
    }
    else if (sort_type == SortType::MERGE_SORT) {
        IndexPairVector buf_vector(data_indexes.size());
        MergeSortSubArray(data_indexes, buf_vector, 0, data_indexes.size(), comparer);
    }
}

void BuildIndexVector(ArrayChunks &data_chunks, IndexPairVector &data_indexes) {
    uint64_t total_size = 0;

    for (uint64_t i = 0; i < data_chunks.size(); i ++) {
        total_size += data_chunks[i].get()->length();
    }

    data_indexes.reserve(total_size);

    for (uint64_t i = 0; i < data_chunks.size(); i ++) {
        for (int64_t j = 0; j < data_chunks[i].get()->length(); j ++) {
            data_indexes.push_back(std::make_pair(i, j));
        }
    }
}

std::shared_ptr<arrow::Array> SortStringIndexes(ArrayChunks &data_chunks,
                                                std::shared_ptr<arrow::DataType> &data_type,
                                                bool ascending,
                                                SortType sort_type) {
    IndexPairVector data_indexes;
    std::vector<uint64_t> acc_counts;

    arrow::Int64Builder result_builder;
    std::shared_ptr<arrow::Array> result_array;

    if (data_type.get()->id() != arrow::Type::STRING) {
        result_builder.Finish(&result_array);
        return result_array;
    }

    BuildIndexVector(data_chunks, data_indexes);

    acc_counts.push_back(0);
    for (size_t i = 0; i < data_chunks.size(); i++) {
        acc_counts.push_back(acc_counts.back() + data_chunks[i].get()->length());
    }

    if (ascending) {
        auto comparer = StringArrayIndexComparer<true>(data_chunks, data_chunks);
        CallSortFunction(data_indexes, sort_type, comparer);
    }
    else {
        auto comparer = StringArrayIndexComparer<false>(data_chunks, data_chunks);
        CallSortFunction(data_indexes, sort_type, comparer);
    }

    result_builder.Reserve(data_indexes.size());
    for (size_t i = 0; i < data_indexes.size(); i++) {
        result_builder.Append(acc_counts[data_indexes[i].first] + data_indexes[i].second);
    }
    result_builder.Finish(&result_array);
    return result_array;
}

std::shared_ptr<arrow::Array> SearchStringIndexes(ArrayChunks &data_chunks,
                                                  std::shared_ptr<arrow::Array> &values,
                                                  std::shared_ptr<arrow::DataType> &data_type,
                                                  std::vector<uint64_t> &sorter,
                                                  BoundSide bound_side) {
    std::vector<uint64_t> result_idxes;
    IndexPairVector data_indexes, *actual_indexes = NULL;

    arrow::Int64Builder result_builder;
    std::shared_ptr<arrow::Array> result_array;

    if (data_type.get()->id() != arrow::Type::STRING) {
        result_builder.Finish(&result_array);
        return result_array;
    }

    BuildIndexVector(data_chunks, data_indexes);

    if (!sorter.size()) {
        actual_indexes = &data_indexes;
    }
    else {
        IndexPairVector data_indexes_new;
        data_indexes_new.reserve(data_indexes.size());
        for (uint64_t i = 0; i < sorter.size(); i++) {
            data_indexes_new.push_back(data_indexes[sorter[i]]);
        }
        actual_indexes = &data_indexes_new;
    }

    ArrayChunks values_chunks { values };
    result_idxes.reserve(values.get()->length());

    for (uint64_t i = 0; i < values.get()->length(); i++) {
        auto value_pair = std::make_pair((uint64_t)0, i);

        if (bound_side == BoundSide::LEFT) {
            auto comparer = StringArrayIndexComparer<true>(data_chunks, values_chunks);
            result_idxes.push_back(
                std::lower_bound(actual_indexes->begin(), actual_indexes->end(), value_pair, comparer)
                - actual_indexes->begin()
            );
        }
        else {
            auto comparer = StringArrayIndexComparer<true>(values_chunks, data_chunks);
            result_idxes.push_back(
                std::upper_bound(actual_indexes->begin(), actual_indexes->end(), value_pair, comparer)
                - actual_indexes->begin()
            );
        }
    }

    result_builder.Reserve(result_idxes.size());
    for (size_t i = 0; i < result_idxes.size(); i++) {
        result_builder.Append(result_idxes[i]);
    }
    result_builder.Finish(&result_array);
    return result_array;
}

}
}
}

#endif
