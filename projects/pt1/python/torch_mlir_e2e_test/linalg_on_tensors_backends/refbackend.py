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
import torch_mlir.dialects.torch
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

from .abc import LinalgOnTensorsBackend

__all__ = [
    "RefBackendLinalgOnTensorsBackend",
]


def assert_arg_type_is_supported(ty):
    SUPPORTED = [
        np.float16, np.float32, np.float64, np.uint8, np.int8, np.int32,
        np.int64, np.bool_, np.complex64, np.complex128
    ]
    assert ty in SUPPORTED, f"Only numpy arrays with dtypes in {SUPPORTED} are supported, but got {ty}"


memref_type_to_np_dtype = {
    "mrf16": np.float16,
    "mrf32": np.float32,
    "mrf64": np.float64,
    "mri1": np.bool_,
    "mri8": np.int8,
    "mri32": np.int32,
    "mri64": np.int64,
    "mrc32": np.complex64,
    "mrc64": np.complex128
}
elemental_type_to_ctype = {
    "i1": ctypes.c_bool,
    "i8": ctypes.c_byte,
    "i64": ctypes.c_int,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double
}

CONSUME_RETURN_FUNC_PREFIX = "refbackend_consume_func_return_"


def get_return_funcs(module):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    return_funcs = []
    with module.context:
        for func in module.body:
            # Returns strings of the form `"refbackend.."` so `"` is deleted.
            func_name = str(func.attributes["sym_name"]).replace('"', '')
            if func_name[:return_prefix_len] == CONSUME_RETURN_FUNC_PREFIX:
                return_funcs.append(func_name)

    return return_funcs


def get_ctype_func(func_name):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    ret_types = func_name[return_prefix_len:].split("_")
    ctypes_arg = [None]
    for type in ret_types:
        if type in elemental_type_to_ctype:
            ctypes_arg.append(elemental_type_to_ctype[type])
        elif type in memref_type_to_np_dtype:
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
        else:
            assert False, f"Not supported type: {type}"

    return ctypes.CFUNCTYPE(*ctypes_arg), ret_types


class RefBackendInvoker:

    def __init__(self, module):
        self.ee = ExecutionEngine(module)
        self.result = None

        return_funcs = get_return_funcs(module)

        for ret_func in return_funcs:
            ctype_wrapper, ret_types = get_ctype_func(ret_func)

            def consume_return_funcs(*args):
                self.result = tuple([
                    arg if type in elemental_type_to_ctype
                    else unranked_memref_to_numpy(
                        arg, memref_type_to_np_dtype[type])
                    for arg, type in zip(args, ret_types)
                ])
                if len(self.result) == 1:
                    self.result = self.result[0]

            self.ee.register_runtime(ret_func,
                                     ctype_wrapper(consume_return_funcs))

    def __getattr__(self, function_name: str):

        def invoke(*args):
            ffi_args = []
            for arg in args:
                assert_arg_type_is_supported(arg.dtype)
                ffi_args.append(
                    ctypes.pointer(
                        ctypes.pointer(get_unranked_memref_descriptor(arg))))

            self.ee.invoke(function_name, *ffi_args)
            result = self.result
            assert result is not None, "Invocation didn't produce a result"
            self.result = None
            return result

        return invoke


LOWERING_PIPELINE = "builtin.module(" + ",".join([
    "func.func(refback-generalize-tensor-pad)",
    "func.func(refback-generalize-tensor-concat)",
    # Apply some optimizations. It would be great if MLIR had more useful
    # optimizations that worked out of the box here.
    # Note: When measured, this doesn't seem to actually help that much
    # for the linalg-on-tensors backend.
    # This is likely because if things are naturally fusable we usually already
    # emit things in that form from the high level (e.g. single linalg-generic).
    # Other backends are likely to benefit more.
    "func.func(linalg-generalize-named-ops)",
    "func.func(linalg-fuse-elementwise-ops)",
    "convert-shape-to-std",
    # MLIR Sparsifier mini-pipeline. Note that this is the bare minimum
    # to ensure operations on sparse tensors are lowered to loops.
    "sparse-assembler",
    "sparsification-and-bufferization",
    "sparse-storage-specifier-to-llvm",
    "inline",  # inline sparse helper methods where useful
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(tm-tensor-bufferize)",
    "func.func(empty-tensor-to-alloc-tensor)",
    "func.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "refback-mlprogram-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "func.func(buffer-deallocation)",
    # Munge to make it ExecutionEngine compatible.
    # Specifically, we rewrite calling convention boundaries to be in terms
    # of unranked memref, and we rewrite the return to actually be a
    # callback that consumes the return (the final munged function always
    # returns void at the C level -- we get the return value by providing the
    # callback).
    "refback-munge-calling-conventions",
    # Insert global variable and instruction sequence for getting the next
    # global seed used in stateful rng.
    # Lower to LLVM
    "func.func(tm-tensor-to-loops)",
    "func.func(refback-munge-memref-copy)",
    "func.func(convert-linalg-to-loops)",
    "func.func(lower-affine)",
    "convert-scf-to-cf",
    "func.func(refback-expand-ops-for-llvm)",
    "func.func(arith-expand)",
    "func.func(convert-math-to-llvm)",
    # Handle some complex mlir::math ops (e.g. atan2)
    "convert-math-to-libm",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "lower-affine",
    "convert-bufferization-to-memref",
    "finalize-memref-to-llvm",
    "func.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "convert-complex-to-llvm",
    "reconcile-unrealized-casts",
]) + ")"


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
            "Lowering Linalg-on-Tensors IR to LLVM with RefBackend",
            enable_ir_printing=False,
        )
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        return RefBackendInvoker(module)
