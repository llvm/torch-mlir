# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import inspect
import re
from typing import List, Optional, Union, Any, Dict
import codecs

import torch

from torch_mlir.jit_ir_importer import ModuleBuilder
from torch_mlir.passmanager import PassManager

from .registry import Registry


def all_integer_dtypes() -> List[int]:
    return [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def is_integer_dtype(dtype: int) -> bool:
    return dtype in all_integer_dtypes()


def all_complex_dtypes() -> List[int]:
    return [torch.complex64, torch.complex128]


def is_complex_dtype(dtype: int) -> bool:
    return dtype in all_complex_dtypes()


def all_float_dtypes() -> List[int]:
    return [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def is_float_dtype(dtype: int) -> bool:
    return dtype in all_float_dtypes()


def get_priority_of_dtype(dtype: int) -> int:
    # If a loop is used to iterate over a list of sorted dtypes, TorchScript
    # produces a loop with INT64_MAX max trip count, which causes problems
    # during the loop unrolling that takes place when simplifying the dtype
    # functions. Therefore, here we resort to `if`s.
    if dtype == torch.bool:
        return 0
    if dtype == torch.uint8:
        return 1
    if dtype == torch.int8:
        return 2
    if dtype == torch.int16:
        return 3
    if dtype == torch.int32:
        return 4
    if dtype == torch.int64:
        return 5
    if dtype == torch.bfloat16:
        return 6
    if dtype == torch.float16:
        return 7
    if dtype == torch.float32:
        return 8
    if dtype == torch.float64:
        return 9
    if dtype == torch.complex64:
        return 10
    if dtype == torch.complex128:
        return 11
    assert False, "Cannot determine priority of dtype"


def get_dtype_of_scalar(scalar: Union[int, float, complex]) -> int:
    # This is hacky. `NumToTensor` is the only PyTorch op for scalars
    # that when `jit.script`ed converts a float scalar to a tensor
    # with dtype that corresponds to Python's `float`.
    #
    # See definition of `NumToTensor`: https://github.com/pytorch/pytorch/blob/c09929659ce8ba2f1b7b2f6e50084ccbf854d44b/torch/csrc/jit/ir/ir.cpp#L1850
    # Definition of `fromNumberType` used by `NumToTensor` to
    # calculate dtype: https://github.com/pytorch/pytorch/blob/c09929659ce8ba2f1b7b2f6e50084ccbf854d44b/aten/src/ATen/core/jit_type.h#L1679
    #
    # Note that doing something like
    # `torch.tensor(scalar).to(type(scalar)).dtype` does not work because
    # `torch.tensor` needs to know the type of the input at compile time
    # and there is no `jit.script` support for Python's `type`.
    #
    # TODO: A better way to handle this would be to add support for
    # `isinstance` in torch-mlir, which requires adding a new torch dialect
    # op.
    return torch.ops.prim.NumToTensor(scalar).dtype


# When we import into torch-mlir, only the calls to
# `__torch_mlir_internal_promote_dtypes` are used to generate the
# `torch.promote_dtypes` ops. Therefore, to avoid generating extra
# MLIR code in the library, all calls made inside
# `__torch_mlir_internal_promote_dtypes` are `jit.ignore`d.
@torch.jit.ignore
def _get_scalar_with_dtype(dtype: torch.dtype) -> Union[int, float]:
    if dtype == torch.int64:
        return 0
    elif dtype == torch.float64:
        return 0.0
    else:
        raise ValueError(f"Unhandled dtype: {dtype}")


@torch.jit.ignore
def _promote_scalar_tensor(
    scalar_dtype: torch.dtype, tensor_rank: int, tensor_dtype: torch.dtype
) -> torch.dtype:
    scalar = _get_scalar_with_dtype(scalar_dtype)
    tensor = torch.rand([1] * tensor_rank).to(tensor_dtype)
    return torch.result_type(scalar, tensor)


@torch.jit.ignore
def _promote_tensor_tensor(
    lhs_rank: int, lhs_dtype: torch.dtype, rhs_rank: int, rhs_dtype: torch.dtype
) -> torch.dtype:
    lhs_tensor = torch.rand([1] * lhs_rank).to(lhs_dtype)
    rhs_tensor = torch.rand([1] * rhs_rank).to(rhs_dtype)
    return torch.result_type(lhs_tensor, rhs_tensor)


@torch.jit.ignore
def _promote_scalar_scalar(
    lhs_dtype: torch.dtype, rhs_dtype: torch.dtype
) -> torch.dtype:
    # When `torch.result_type` is used on two scalars, the result
    # dtype is the dtype one would expect for an op with signature
    # (Scalar, Scalar) -> (Tensor). However, once a module gets
    # jit.scripted, all math ops that work on scalars becomes
    # (Scalar, Scalar) -> (Scalar) ops. So to get the right result
    # dtype, we use the tensor-tensor promotion rules.
    return _promote_tensor_tensor(0, lhs_dtype, 0, rhs_dtype)


def promote_dtypes(
    ranks: List[Optional[int]], dtypes: List[torch.dtype]
) -> torch.dtype:
    """Apply PyTorch dtype promotion rules and return the result type."""
    return __torch_mlir_internal_promote_dtypes(ranks, dtypes)


def __torch_mlir_internal_promote_dtypes(
    ranks: List[Optional[int]], dtypes: List[torch.dtype]
) -> torch.dtype:
    """Apply PyTorch dtype promotion rules and return the result type.

    This function serves two purposes:
    1. It is handled in a special way during import into Torch-MLIR,
       generating `torch.promote_dtypes` ops
    2. Computes the actual promotion logic at the Python level in order
       to be able to test dtype calculation functions against PyTorch
    """
    lhs_optional_rank = ranks[0]
    lhs_dtype = dtypes[0]
    for rhs_optional_rank, rhs_dtype in zip(ranks, dtypes):
        if lhs_optional_rank is None and rhs_optional_rank is None:
            lhs_dtype = _promote_scalar_scalar(lhs_dtype, rhs_dtype)
        elif lhs_optional_rank is None and rhs_optional_rank is not None:
            lhs_dtype = _promote_scalar_tensor(lhs_dtype, rhs_optional_rank, rhs_dtype)
            lhs_optional_rank = rhs_optional_rank
        elif lhs_optional_rank is not None and rhs_optional_rank is None:
            lhs_dtype = _promote_scalar_tensor(rhs_dtype, lhs_optional_rank, lhs_dtype)
        elif lhs_optional_rank is not None and rhs_optional_rank is not None:
            lhs_dtype = _promote_tensor_tensor(
                lhs_optional_rank, lhs_dtype, rhs_optional_rank, rhs_dtype
            )
            lhs_optional_rank = max(lhs_optional_rank, rhs_optional_rank)
    return lhs_dtype


def not_present_in_registry(f):
    """Decorator for abstract interpretation functions not present in the registry.

    This can happen for "valsem" ops that we have in Torch-MLIR, such as
    torch.valsem.aten.fill.Scalar, which are consistent with PyTorch conventions
    (e.g. being the value-semantic correspondent of torch.aten.fill_.Scalar),
    but that for whatever reason are not present in PyTorch. Such ops are useful
    to keep certain passes within Torch-MLIR as consistent as possible.
    For such ops, in the shape library generator, we treat them as if they
    were registered torch ops (so we don't put "valsem" on them), to keep the
    generator consistent.

    To check if this decorator has been applied, use
    `hasattr(f, "_not_present_in_registry")`.
    """
    f._not_present_in_registry = None
    return f


def _verify_signature_matches_registry(f, registry: Registry):
    source = inspect.getsource(f)
    signature = None
    for line in source.splitlines():
        if line.startswith("def "):
            signature = line
            break
    assert signature is not None, f"Could not find signature for {f.__name__}"
    assert (
        "〡" in signature
    ), f"Malformed signature {signature}. Signature missing the character `〡`"
    function_name, function_kind = f.__name__.split("〡")
    atoms = function_name.split("〇")
    if len(atoms) == 2:
        atoms += [""]
    try:
        operator = registry.get_by_triple(tuple(atoms))
    except KeyError as e:
        raise ValueError(f"Unable to find op {'.'.join(atoms)} in registry")
    if function_kind == "shape":
        expected_signature = operator.get_shape_function_signature()
    elif function_kind == "dtype":
        expected_signature = operator.get_dtype_function_signature()
    elif function_kind == "decomposition":
        expected_signature = operator.get_decomposition_function_signature()
    elif function_kind == "has_value_semantics":
        expected_signature = operator.get_has_value_semantics_function_signature()
    else:
        raise ValueError(f"Invalid Op signature function kind: '{function_kind}'")
    if signature != expected_signature:
        raise ValueError(
            f"Signature mismatch for {f.__name__!r}: expected {expected_signature!r}, got {signature!r}"
        )


def generate_library(functions: Dict[str, Any]) -> str:
    """Convert all op functions in `functions` into MLIR."""
    mb = ModuleBuilder()
    # We use the registry to ensure that the shape functions are consistent
    # with the ops.
    registry = Registry.load()
    for k, v in functions.items():
        if "〇" not in k:
            continue
        if not hasattr(v, "_not_present_in_registry"):
            _verify_signature_matches_registry(v, registry)
        # Add it to the compilation unit.
        torch.jit.script(v)
    for function in torch.jit._state._python_cu.get_functions():
        # Calls to the function `__torch_mlir_internal_promote_dtypes`
        # will get converted to the torch-dialect op `torch.promote_dtypes`
        # during import, so there is no need to import the actual
        # function.
        if function.name == "__torch_mlir_internal_promote_dtypes":
            continue
        mb.import_function(function)
    # Clean up the IR a bit before writing it out.
    pm = PassManager.parse("builtin.module(canonicalize)", context=mb.module.context)
    pm.run(mb.module.operation)
    # Munge the IR a bit to make it more systematically accessible.
    asm = mb.module.operation.get_asm()
    # We'd like a unique function prefix to avoid collisions with user-
    # defined symbols. Since all of our shape functions conveniently have
    # a `〇` in them, we replace the torch namespace with our prefix. E.g.:
    # __torch__.aten〇add〇Scalar -> __torch_mlir_shape_fn.aten〇add〇Scalar

    # Encoding for: 〇
    circle = r"\\E3\\80\\87"
    # Encoding for: 〡
    line = r"\\E3\\80\\A1"
    name = r"[^.(]+"
    # Sometimes PyTorch will insert namespaces to the function name in
    # the format: `__torch__.{namespace_1}.{namespace_2}...{op_name}`
    # The extra namespaces are not part of the abstract interpretation
    # function name, so here we simply drop the extra namespaces.
    namespace = rf"(?:{name}\.)"

    asm = re.sub(
        rf'@"__torch__\.{namespace}*({name}){circle}({name}){line}({name})"',
        rf'@"__torch_mlir_\3_fn.\1{circle}\2"',
        asm,
    )

    # Put the `〇` back to a regular `.`.
    asm = asm.replace(codecs.decode(circle, "unicode_escape"), ".")
    return asm
