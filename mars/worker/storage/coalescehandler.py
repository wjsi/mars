from .core import StorageHandler, BytesStorageMixin, DataStorageDevice, \
    register_storage_handler_cls


class CoalesceHandler(StorageHandler, BytesStorageMixin):
    storage_type = DataStorageDevice.COALESCE

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        return DiskIO(session_id, data_key, 'r', packed=packed, compress=packed_compression, handler=self)

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, auto_register=True, pin_token=None,
                            _promise=False):
        return DiskIO(session_id, data_key, 'w', total_bytes, compress=self._compress,
                      packed=packed, handler=self)

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _fallback(*_):
            return promise.all_(
                src_handler.create_bytes_reader(session_id, k, _promise=True)
                    .then(lambda reader: self.create_bytes_writer(
                    session_id, k, reader.nbytes, _promise=True)
                          .then(lambda writer: self._copy_bytes_data(reader, writer),
                                lambda *exc: self.pass_on_exc(reader.close, exc)))
                for k in data_keys)

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        data_dict = dict()

        def _load_single_data(key):
            data_size = self._get_serialized_data_size(data_dict[key])
            return self.create_bytes_writer(session_id, key, data_size, _promise=True) \
                .then(lambda writer: self._copy_object_data(data_dict.pop(key), writer),
                      lambda *exc: self.pass_on_exc(functools.partial(data_dict.pop, key), exc))

        def _load_all_data(objs):
            data_dict.update(zip(data_keys, objs))
            objs[:] = []
            return promise.all_(_load_single_data(k) for k in data_keys) \
                .catch(lambda *exc: self.pass_on_exc(data_dict.clear, exc))

        def _fallback(*_):
            return src_handler.get_objects(session_id, data_keys, serialize=True, _promise=True) \
                .then(_load_all_data, lambda *exc: self.pass_on_exc(data_dict.clear, exc))

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def delete(self, session_id, data_keys, _tell=False):
        pass


register_storage_handler_cls(DataStorageDevice.CUDA, CoalesceHandler)
