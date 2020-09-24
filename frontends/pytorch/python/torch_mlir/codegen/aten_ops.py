#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Populates an op registry for ATen ops.

Typically callers will import and use the 'populate' function to add known
ops to the OpRegistry. When run interactively as a main module, it simply
prints all registered ops.
"""

from .registry import *

import torch
import torch.nn.functional as F


def populate(r: OpRegistry):
  # Unary pointwise ops (ordinary that take out refs).
  for f in [
      torch.abs, torch.acos, torch.angle, torch.asin, torch.atan, torch.ceil,
      torch.conj, torch.cos, torch.cosh, torch.digamma, torch.erf, torch.erfc,
      torch.erfinv, torch.exp, torch.expm1, torch.floor, torch.frac,
      torch.lgamma, torch.log, torch.log10, torch.log1p, torch.log2, torch.neg,
      torch.reciprocal, torch.round, torch.rsqrt, torch.sigmoid, torch.sign,
      torch.sin, torch.sinh, torch.sqrt, torch.tan, torch.tanh, torch.trunc
  ]:
    r.op(f, TensorValue("input")).with_outref_variant()

  # Unary pointwise ops (that do not take out refs).
  for f in [torch.relu]:
    r.op(f, TensorValue("input"))

  # Binary pointwise ops.
  r.op(torch.add,
       TensorValue("input"),
       TensorValue("other"),
       alpha=LiteralValue()).with_outref_variant()
  r.op(torch.atan2, TensorValue("input"),
       TensorValue("other")).with_outref_variant()
  r.op(torch.div, TensorValue("input"),
       TensorValue("other")).with_outref_variant()
  r.op(torch.floor_divide, TensorValue("input"),
       TensorValue("other")).with_outref_variant()
  r.op(torch.mul, TensorValue("input"),
       TensorValue("other")).with_outref_variant()
  r.op(torch.remainder, TensorValue("input"),
       TensorValue("other")).with_outref_variant()
  r.op(torch.true_divide, TensorValue("dividend"),
       TensorValue("divisor")).with_outref_variant()

  # Aggregate operations.
  # TODO: Support the optional dtype= parameter.
  r.op(torch.cumsum, TensorValue("input", example_size=(10, 3)),
       LiteralValue("dim", value=1)).with_outref_variant()
  r.op(
      torch.mean,
      TensorValue("input", example_size=(10, 3)),  #
      LiteralValue("dim", value=[0], mlir_ods_predicate="ATen_IntList"),  #
      LiteralValue("keep_dim",
                   value=False,
                   mlir_ods_predicate="ATen_BoolScalar")  #
  ).with_outref_variant()
  r.op(
      torch.sum,
      TensorValue("input", example_size=(10, 3)),  #
      LiteralValue("dim", value=[0], mlir_ods_predicate="ATen_IntList"),  #
      LiteralValue("keep_dim",
                   value=False,
                   mlir_ods_predicate="ATen_BoolScalar")  #
  ).with_outref_variant()

  # Gather.
  (r.op(
      torch.gather,  #
      TensorValue("input", example_size=(2, 2)),  #
      dim=LiteralValue(value=1, mlir_ods_predicate="ATen_IntScalar"),  #
      index=LiteralValue(value=torch.tensor([[0, 0], [1, 0]]),
                         mlir_ods_predicate="ATen_AnyTensor"),  #
      sparse_grad=LiteralValue(value=False,
                               mlir_ods_predicate="ATen_BoolScalar")  #
  ).with_outref_variant())

  # Non-view methods on Tensor.
  (r.op((TensorValue(name="input", example_size=(3, 1)), "@T")))

  # BLAS and LAPACK ops.
  r.op(torch.addmm,
       TensorValue("input", example_size=(2, 3)),
       TensorValue("mat1", example_size=(2, 3)),
       TensorValue("mat2", example_size=(3, 3)),
       beta=LiteralValue(),
       alpha=LiteralValue()).with_outref_variant()
  r.op(torch.dot, TensorValue("input", example_size=(10,)),
       TensorValue("tensor", example_size=(10,)))
  r.op(torch.matmul, TensorValue("input", example_size=(10, 3, 4)),
       TensorValue("other", example_size=(4, 5))).with_outref_variant()
  r.op(torch.mm, TensorValue("input", example_size=(3, 4)),
       TensorValue("mat2", example_size=(4, 6))).with_outref_variant()

  # NN Functional.
  # Note that _convolution is a special case and is manually coded.
  r.op(
      F.hardtanh,
      TensorValue("input"),  #
      min_val=LiteralValue(value=-1.0,
                           mlir_ods_predicate="ATen_FloatScalar"),  #
      max_val=LiteralValue(value=1.0, mlir_ods_predicate="ATen_FloatScalar")  #
  )
  r.op(F.avg_pool1d,
       TensorValue("input", example_size=(1, 1, 7)),
       kernel_size=LiteralValue(value=[3], mlir_ods_predicate="ATen_IntList"),
       stride=LiteralValue(value=[5], mlir_ods_predicate="ATen_IntList"),
       padding=LiteralValue(value=[1], mlir_ods_predicate="ATen_IntList"),
       ceil_mode=LiteralValue(value=True, mlir_ods_predicate="ATen_BoolScalar"),
       count_include_pad=LiteralValue(value=False,
                                      mlir_ods_predicate="ATen_BoolScalar"))

  # MaxPool1D is split into two ops based on whether return_indices is True:
  #   aten::max_pool1d -> tensor<float>
  #   aten::max_pool1d_with_indices -> (tensor<float>, tensor<long>)
  # Both have odd signatures and are hand-mapped.
  # TODO: Implement max_pool1d(..., with_indices=True)
  (r.op(F.max_pool1d,
        TensorValue("input", example_size=(1, 1, 7)),
        kernel_size=LiteralValue(value=[3], mlir_ods_predicate="ATen_IntList"),
        stride=LiteralValue(value=[5], mlir_ods_predicate="ATen_IntList"),
        padding=LiteralValue(value=[1], mlir_ods_predicate="ATen_IntList"),
        dilation=LiteralValue(value=[3], mlir_ods_predicate="ATen_IntList"),
        ceil_mode=LiteralValue(value=True,
                               mlir_ods_predicate="ATen_BoolScalar"),
        return_indices=LiteralValue(value=False))  #
   .with_torch_op_kind("aten::max_pool1d")  #
   .with_operand_map("input", "kernel_size", "stride", "padding", "dilation",
                     "ceil_mode")  #
  )

  # View ops.
  # TODO: All of these need special analysis and should be parameterized
  # on mutable tensors and have a proper design thought through. For now,
  # even having them in the inventory (badly) increases visibility.
  (r.op(torch.as_strided,
        TensorValue("input"),
        size=LiteralValue(value=[2, 2], mlir_ods_predicate="ATen_IntList"),
        stride=LiteralValue(value=[1, 2], mlir_ods_predicate="ATen_IntList"),
        storage_offset=LiteralValue(value=4,
                                    mlir_ods_predicate="ATen_IntScalar"))  #
   .with_append_description(r"""

     MLIR Specific Notes
     -------------------
     In PyTorch proper, this op creates a view that may internally alias. And
     have explicit warnings about avoiding inplace updates on such a
     view (without first cloning). For the moment, this op is formulated with
     value semantics that imply a copy instead of a view, and it is expected
     that any sharing can be recovered later by the compiler. The warning
     about not in-place updating of such a result should be treated as UB
     when compiled.
   """))

  (r.op((TensorValue(name="input", example_size=(3, 1)), "expand"),
        LiteralValue("sizes", value=torch.Size([3, 4])))  #
   .with_operand_map("input", "sizes",
                     LiteralValue("implicit",
                                  mlir_ods_predicate="ATen_BoolScalar"))  #
   .with_append_description(r"""

     MLIR Specific Notes
     -------------------
     See notes for the 'as_strided' op.
     """))

  (r.op(
      torch.squeeze,
      TensorValue("input"),  #
      LiteralValue("dim", value=1, mlir_ods_predicate="ATen_IntScalar")  #
  ).with_append_description(r"""

     MLIR Specific Notes
     -------------------
     See notes for the 'as_strided' op.
     """))

  (r.op((TensorValue(name="input", example_size=(3, 1)), "view"),
        LiteralValue(name="size",
                     value=[3, 1],
                     mlir_ods_predicate="ATen_IntList"))
   .with_append_description(r"""

     MLIR Specific Notes
     -------------------
     See notes for the 'as_strided' op.
     """))


if __name__ == "__main__":
  import logging
  logging.basicConfig(level=logging.DEBUG)
  registry = OpRegistry()
  populate(registry)
  print("Registered operations:")
  for m in registry.mappings:
    print(" ", m)
