import ctypes
import multiprocessing as mp
import os

import numpy as np


class SharedMem:
    """Multiprocess shared memory for statistical/performance data."""

    _mem = {}
    _shape = {}

    @classmethod
    def add(cls, name, array, shape):
        cls._mem[name] = array
        cls._shape[name] = shape

    @classmethod
    def to_share(cls):
        return cls._mem, cls._shape

    @classmethod
    def share_with(cls, mem, shape):
        cls._mem = mem
        cls._shape = shape

    @classmethod
    def dump(cls, name, index, data):
        arr = cls._shm_to_arr(cls._mem[name], cls._shape[name])
        np.copyto(dst=arr[index], src=data)

    @classmethod
    def _shm_to_arr(cls, raw_array, shape):
        dtype_dict = {
            ctypes.c_double: np.float64,
            ctypes.c_int: np.int32
        }
        return np.frombuffer(raw_array,
                             dtype=dtype_dict[raw_array._type_]).reshape(shape)

    @classmethod
    def get(cls):
        return {
            name: cls._shm_to_arr(cls._mem[name], cls._shape[name])
            for name in cls._mem
        }


class Sink:

    def __init__(self, cfg, filename):
        self.cfg = cfg
        self.filename = filename

    def __enter__(self):
        for name in self.cfg:
            c = self.cfg[name]
            shape = (c[0], c[1])
            array = mp.RawArray(c[2], shape[0] * shape[1])

            SharedMem.add(name, array, shape)

        return SharedMem

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            return

        store = Store(self.filename)
        mem = SharedMem.get()

        for name in mem:
            store.replace(name, mem[name])

        store.close()


class Store:
    """Wrapper for access to hdf5 data file."""

    def __init__(self, filename, f_key=None):
        import h5py

        dump_dir = r'.dump'
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        self.hf = h5py.File(filename, 'a')
        self.f_key = f_key

    def replace(self, key, value):
        hf_key = self._key(key)
        if hf_key in self.hf.keys():
            del self.hf[hf_key]
        self.hf.create_dataset(hf_key, data=value, compression='gzip',
                               chunks=True)

    def _key(self, key):
        if self.f_key:
            hf_key = self.f_key(key)
        else:
            hf_key = key
        return hf_key

    def __getitem__(self, key):
        hf_key = self._key(key)
        return self.hf[hf_key]

    def __contains__(self, key):
        return self._key(key) in self.hf.keys()

    def close(self):
        self.hf.close()
