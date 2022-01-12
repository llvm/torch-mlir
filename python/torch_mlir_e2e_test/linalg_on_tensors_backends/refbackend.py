# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ctypes
import numpy as np

from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.execution_engine import *
from torch_mlir.runtime import *
# Imported for side effects.
import torch_mlir.all_passes_registration
import torch_mlir.dialects.torch

from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report

from .abc import LinalgOnTensorsBackend

__all__ = [
    "RefBackendLinalgOnTensorsBackend",
]


def checkArgTypeIsSupported(ty):
    SUPPORTED = [np.float32, np.float64, np.int32, np.int64, np.bool_]
    assert ty in SUPPORTED, f"Only numpy arrays with dtypes in {SUPPORTED} are supported"


class RefBackendInvoker:
    def __init__(self, module):
        self.ee = ExecutionEngine(module)
        self.result = None

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri1(a):
            self.result = unranked_memref_to_numpy(a, np.bool_)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri32(a):
            self.result = unranked_memref_to_numpy(a, np.int32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mri64(a):
            self.result = unranked_memref_to_numpy(a, np.int64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32(a):
            self.result = unranked_memref_to_numpy(a, np.float32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf64(a):
            self.result = unranked_memref_to_numpy(a, np.float64)

        @ctypes.CFUNCTYPE(None, ctypes.c_int)
        def consume_return_i64(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.c_float)
        def consume_return_f32(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.c_double)
        def consume_return_f64(a):
            self.result = a

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mri64(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.int64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mrf32(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.float32)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf64_mrf64(arg0, arg1):
            self.result = unranked_memref_to_numpy(
                arg0, np.float64), unranked_memref_to_numpy(
                    arg1,
                    np.float64)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor),
                          ctypes.POINTER(UnrankedMemRefDescriptor))
        def consume_return_mrf32_mrf32_mrf32(arg0, arg1, arg2):
            self.result = unranked_memref_to_numpy(
                arg0, np.float32), unranked_memref_to_numpy(
                    arg1,
                    np.float32), unranked_memref_to_numpy(arg2, np.float32)

        self.ee.register_runtime("refbackend_consume_func_return_mri1",
                                 consume_return_mri1)

        self.ee.register_runtime("refbackend_consume_func_return_mri32",
                                 consume_return_mri32)

        self.ee.register_runtime("refbackend_consume_func_return_mri64",
                                 consume_return_mri64)

        self.ee.register_runtime("refbackend_consume_func_return_mrf32",
                                 consume_return_mrf32)

        self.ee.register_runtime("refbackend_consume_func_return_mrf64",
                                 consume_return_mrf64)

        self.ee.register_runtime("refbackend_consume_func_return_i64",
                                 consume_return_i64)

        self.ee.register_runtime("refbackend_consume_func_return_f32",
                                 consume_return_f32)

        self.ee.register_runtime("refbackend_consume_func_return_f64",
                                 consume_return_f64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mri64",
            consume_return_mrf32_mri64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mrf32",
            consume_return_mrf32_mrf32)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf64_mrf64",
            consume_return_mrf64_mrf64)

        self.ee.register_runtime(
            "refbackend_consume_func_return_mrf32_mrf32_mrf32",
            consume_return_mrf32_mrf32_mrf32)

    def __getattr__(self, function_name: str):
        def invoke(*args):
            ffi_args = []
            for arg in args:
                checkArgTypeIsSupported(arg.dtype)
                ffi_args.append(
                    ctypes.pointer(
                        ctypes.pointer(get_unranked_memref_descriptor(arg))))

            self.ee.invoke(function_name, *ffi_args)
            result = self.result
            assert result is not None, "Invocation didn't produce a result"
            self.result = None
            return result

        return invoke


LOWERING_PIPELINE = ",".join([
    # Bufferize.
    "tensor-constant-bufferize",
    "builtin.func(scf-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    "builtin.func(finalizing-bufferize)",
    # Munge to make it ExecutionEngine compatible.
    # Specifically, we rewrite calling convention boundaries to be in terms
    # of unranked memref, and we rewrite the return to actually be a
    # callback that consumes the return (the final munged function always
    # returns void at the C level -- we get the return value by providing the
    # callback).
    "refback-munge-calling-conventions",
    # Insert global variable and instruction sequence for getting the next
    # global seed used in stateful rng.
    "refback-insert-rng-globals",
    # Lower to LLVM
    "builtin.func(convert-linalg-to-loops)",
    "builtin.func(lower-affine)",
    "builtin.func(convert-scf-to-std)",
    "builtin.func(refback-expand-ops-for-llvm)",
    "builtin.func(arith-expand)",
    "builtin.func(convert-math-to-llvm)",
    "convert-memref-to-llvm",
    "convert-std-to-llvm",
    "reconcile-unrealized-casts",
])


class RefBackendLinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""
    def __init__(self):
        super().__init__()

    def compile(self, imported_module: Module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        run_pipeline_with_repro_report(
            imported_module, LOWERING_PIPELINE,
            "Lowering Linalg-on-Tensors IR to LLVM with RefBackend")
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        return RefBackendInvoker(module)
