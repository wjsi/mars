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

import pickle
import base64

import numpy as np
import pandas as pd

from .... import opcodes as OperandDef
from ....serialize import KeyField, StringField
from ....dataframe.core import SERIES_CHUNK_TYPE, DATAFRAME_CHUNK_TYPE
from ....dataframe.utils import parse_index
from ....tensor.core import TENSOR_TYPE, TensorOrder
from ....utils import register_tokenizer, to_str
from ...operands import LearnOperand, LearnOperandMixin, OutputType
from .dmatrix import ToDMatrix, check_data

try:
    from xgboost import Booster

    register_tokenizer(Booster, pickle.dumps)
except ImportError:
    pass


def _on_serialize_model(m):
    return to_str(base64.b64encode(pickle.dumps(m)))


def _on_deserialize_model(ser):
    return pickle.loads(base64.b64decode(ser))


class XGBPredict(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.XGBOOST_PREDICT

    _data = KeyField('data')
    _model = StringField('model', on_serialize=_on_serialize_model, on_deserialize=_on_deserialize_model)

    def __init__(self, data=None, model=None, output_types=None, gpu=None, **kw):
        super(XGBPredict, self).__init__(_data=data, _model=model, _gpu=gpu,
                                         _output_types=output_types, **kw)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    def _set_inputs(self, inputs):
        super(XGBPredict, self)._set_inputs(inputs)
        self._data = self._inputs[0]

    def __call__(self):
        num_class = self._model.attr('num_class')
        if num_class is not None:
            num_class = int(num_class)
        if num_class is not None:
            shape = (len(self._data), int(self._model.attr('num_class')))
        else:
            shape = (len(self._data),)
        if self._output_types[0] == OutputType.tensor:
            return self.new_tileable([self._data], shape=shape, dtype=np.dtype(np.float32),
                                     order=TensorOrder.C_ORDER)
        elif self._output_types[0] == OutputType.dataframe:
            # dataframe
            dtypes = pd.DataFrame(np.random.rand(0, num_class), dtype=np.float32).dtypes
            return self.new_tileable([self._data], shape=shape, dtypes=dtypes,
                                     columns_value=parse_index(dtypes.index),
                                     index_value=self._data.index_value)
        else:
            # series
            return self.new_tileable([self._data], shape=shape, index_value=self._data.index_value,
                                     name='predictions', dtype=np.dtype(np.float32))

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        out_chunks = []
        data = op.data
        if data.chunk_shape[1] > 1:
            data = data.rechunk({1: op.data.shape[1]}).single_tiles()
        for in_chunk in data.chunks:
            chunk_op = op.copy().reset_key()
            chunk_index = (in_chunk.index[0],)
            if op.model.attr('num_class'):
                chunk_shape = (len(in_chunk), 2)
                chunk_index += (0,)
            else:
                chunk_shape = (len(in_chunk),)
            if op.output_types[0] == OutputType.tensor:
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtype=out.dtype,
                                               order=out.order, index=chunk_index)
            elif op.output_types[0] == OutputType.dataframe:
                # dataframe chunk
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtypes=data.dtypes,
                                               columns_value=data.columns,
                                               index_value=in_chunk.index_value,
                                               index=chunk_index)
            else:
                # series chunk
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtype=out.dtype,
                                               index_value=in_chunk.index_value,
                                               name=out.name, index=chunk_index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        nsplits = (data.nsplits[0],)
        if out.ndim > 1:
            nsplits += ((out.shape[1],),)
        params['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        from xgboost import DMatrix

        raw_data = data = ctx[op.data.key]
        if isinstance(data, tuple):
            data = ToDMatrix.get_xgb_dmatrix(data)
        else:
            data = DMatrix(data)
        result = op.model.predict(data)

        if isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE):
            result = pd.DataFrame(result, index=raw_data.index)
        elif isinstance(op.outputs[0], SERIES_CHUNK_TYPE):
            result = pd.Series(result, index=raw_data.index, name='predictions')

        ctx[op.outputs[0].key] = result


def predict(model, data, session=None, run_kwargs=None, run=True):
    from xgboost import Booster

    data = check_data(data)
    if not isinstance(model, Booster):
        raise TypeError('model has to be a xgboost.Booster, got {0} instead'.format(type(model)))

    num_class = model.attr('num_class')
    if isinstance(data, TENSOR_TYPE):
        output_types = [OutputType.tensor]
    elif num_class is not None:
        output_types = [OutputType.dataframe]
    else:
        output_types = [OutputType.series]

    op = XGBPredict(data=data, model=model, gpu=data.op.gpu, output_types=output_types)
    result = op()
    if run:
        result.execute(session=session, fetch=False, **(run_kwargs or dict()))
    return result