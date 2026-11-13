# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

# The RefBackend runs the sparsifier mini-pipeline only for modules that
# contain a sparse_tensor dialect op or a sparse tensor type. These checks
# cover each place the classifier looks: op operands and results, block
# arguments, function signatures, and sparse_tensor ops on dense types.

from torch_mlir.ir import Context, Module
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    module_contains_sparse_tensor,
)

CSR = "#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>"


def run(name, asm):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(CSR + "\n" + asm)
        print(f"{name}: {module_contains_sparse_tensor(module)}")


# CHECK: dense: False
run(
    "dense",
    """
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.addf %a, %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
""",
)

# CHECK: sparse_argument: True
run(
    "sparse_argument",
    """
func.func @main(%a: tensor<4x4xf32, #CSR>) -> tensor<4x4xf32> {
  %0 = sparse_tensor.convert %a : tensor<4x4xf32, #CSR> to tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
""",
)

# CHECK: sparse_result: True
run(
    "sparse_result",
    """
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32, #CSR> {
  %0 = sparse_tensor.convert %a : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
  return %0 : tensor<4x4xf32, #CSR>
}
""",
)

# CHECK: sparse_intermediate: True
run(
    "sparse_intermediate",
    """
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = sparse_tensor.convert %a : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
  %1 = sparse_tensor.convert %0 : tensor<4x4xf32, #CSR> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
""",
)

# CHECK: sparse_block_argument: True
run(
    "sparse_block_argument",
    """
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  "test.region"() ({
  ^bb0(%b: tensor<4x4xf32, #CSR>):
    "test.terminator"() : () -> ()
  }) : () -> ()
  return %a : tensor<4x4xf32>
}
""",
)

# CHECK: sparse_private_function: True
run(
    "sparse_private_function",
    """
func.func private @external(tensor<4x4xf32, #CSR>)
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %a : tensor<4x4xf32>
}
""",
)

# CHECK: sparse_op_on_dense_types: True
run(
    "sparse_op_on_dense_types",
    """
func.func @main(%a: tensor<4xf32>) -> tensor<?xf32> {
  %0 = sparse_tensor.convert %a : tensor<4xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}
""",
)

# CHECK: non_builtin_function_type: False
run(
    "non_builtin_function_type",
    """
llvm.func @external(f32) -> f32
func.func @main(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %a : tensor<4x4xf32>
}
""",
)
