"""
Interface with mlip.cpp's plain C API (``libmlipcpp_c.so``, the library behind LAMMPS'
``pair_style mlipcpp``), through ctypes, without metatomic: positions go in as one
float32 host array per step, energy and forces come back the same way.
"""

import ctypes
import os

import numpy as np

from ipi.utils.units import unit_to_internal, unit_to_user
from .dummy import Dummy_driver

__DRIVER_NAME__ = "mlipcpp_c"
__DRIVER_CLASS__ = "MLIPCppCDriver"


class MLIPCppCDriver(Dummy_driver):
    """
    Driver for mlip.cpp models (GGUF) through ``libmlipcpp_c.so``.

    Command-line:
        i-pi-py_driver -m mlipcpp_c -o template=start.xyz,model=pet-mad-s.gguf,lib=libmlipcpp_c.so

    Parameters:
        :param template: string, an ASE-readable structure giving the atomic numbers
        :param model: string, path to the GGUF model
        :param lib: string, path to libmlipcpp_c.so (default: $MLIPCPP_C_LIB)
        :param backend: str, optional, cpu or cuda (default cuda)
    """

    def __init__(self, template, model, lib=None, backend="cuda", *args, **kwargs):
        self.template, self.model_path = template, str(model)
        self.lib_path = lib or os.environ["MLIPCPP_C_LIB"]
        self.backend = {"cpu": 1, "cuda": 2}[str(backend)]
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        import ase.io

        lib = ctypes.CDLL(self.lib_path)
        lib.mlipcpp_model_create.restype = ctypes.c_void_p
        lib.mlipcpp_model_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.mlipcpp_predict_ptr_with_options.argtypes = [ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_void_p] * 6
        lib.mlipcpp_result_get_energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.mlipcpp_result_get_forces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.mlipcpp_get_last_error.restype = ctypes.c_char_p
        lib.mlipcpp_set_backend(self.backend)

        class Options(ctypes.Structure):  # mlipcpp_model_options_t
            _fields_ = [("backend", ctypes.c_int), ("precision", ctypes.c_int), ("cutoff", ctypes.c_float)]

        class PredictOptions(ctypes.Structure):  # mlipcpp_predict_options_t
            _fields_ = [("forces", ctypes.c_bool), ("stress", ctypes.c_bool), ("nc", ctypes.c_bool)]

        options = Options()
        lib.mlipcpp_model_options_default(ctypes.byref(options))
        options.backend = self.backend
        self.lib, self.model = lib, lib.mlipcpp_model_create(ctypes.byref(options))
        if lib.mlipcpp_model_load(self.model, self.model_path.encode()) != 0:
            raise RuntimeError(lib.mlipcpp_get_last_error().decode())
        self.predict_options = PredictOptions(True, False, False)
        self.types = np.ascontiguousarray(ase.io.read(self.template).numbers, dtype=np.int32)
        self.forces = np.zeros((len(self.types), 3), dtype=np.float32)

    def compute_structure(self, cell, pos):
        x = np.ascontiguousarray(unit_to_user("length", "angstrom", pos).reshape(-1, 3), dtype=np.float32)
        result = ctypes.c_void_p()
        status = self.lib.mlipcpp_predict_ptr_with_options(
            self.model, len(self.types), x.ctypes.data, self.types.ctypes.data, None, None,
            ctypes.byref(self.predict_options), ctypes.byref(result))
        if status != 0:
            raise RuntimeError(self.lib.mlipcpp_get_last_error().decode())
        energy = ctypes.c_float()
        self.lib.mlipcpp_result_get_energy(result, ctypes.byref(energy))
        self.lib.mlipcpp_result_get_forces(result, self.forces.ctypes.data, len(self.types))
        return (unit_to_internal("energy", "electronvolt", float(energy.value)),
                unit_to_internal("force", "ev/ang", self.forces.astype(np.float64)).reshape(np.shape(pos)),
                np.zeros((3, 3)), "")
