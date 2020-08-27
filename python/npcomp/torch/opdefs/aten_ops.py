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

  # Binary pointwise ops.
  r.op(torch.add,
       TensorValue("input"),
       TensorValue("other"),
       alpha=ScalarValue()).with_outref_variant()
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

  # Other operations.
  # TODO: Support the optional dtype= parameter.
  r.op(torch.cumsum, TensorValue("input", example_size=(10, 3)),
       ScalarValue("dim", value=1)).with_outref_variant()

  # BLAS and LAPACK ops.
  r.op(torch.addmm,
       TensorValue("input", example_size=(2, 3)),
       TensorValue("mat1", example_size=(2, 3)),
       TensorValue("mat2", example_size=(3, 3)),
       beta=ScalarValue(),
       alpha=ScalarValue()).with_outref_variant()
  r.op(torch.dot, TensorValue("input", example_size=(10,)),
       TensorValue("tensor", example_size=(10,)))
  r.op(torch.matmul, TensorValue("input", example_size=(10, 3, 4)),
       TensorValue("other", example_size=(4, 5))).with_outref_variant()
  r.op(torch.mm, TensorValue("input", example_size=(3, 4)),
       TensorValue("mat2", example_size=(4, 6))).with_outref_variant()

  # NN Functional.
  # Note that _convolution is a special case and is manually coded.
  r.op(F.avg_pool1d,
       TensorValue("input", example_size=(1, 1, 7)),
       kernel_size=ScalarValue(value=[3]),
       stride=ScalarValue(value=[5]),
       padding=ScalarValue(value=[1]),
       ceil_mode=ScalarValue(value=True),
       count_include_pad=ScalarValue(value=False))
  r.op(F.max_pool1d,
       TensorValue("input", example_size=(1, 1, 7)),
       kernel_size=ScalarValue(value=[3]),
       stride=ScalarValue(value=[5]),
       padding=ScalarValue(value=[1]),
       ceil_mode=ScalarValue(value=True))


if __name__ == "__main__":
  import logging
  logging.basicConfig(level=logging.DEBUG)
  registry = OpRegistry()
  populate(registry)
  print("Registered operations:")
  for m in registry.mappings:
    print(" ", m)
