# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# Verify without debug info.
# CHECK-LABEL: func @add3
# CHECK-SAME: (%arg0: !numpy.ndarray<*:!numpy.any_dtype>, %arg1: !numpy.ndarray<*:!numpy.any_dtype>, %arg2: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
# CHECK:   %[[C1:.*]] = constant 1 : i64
# CHECK:   %[[A0:.*]] = torch.kernel_call "aten::add" %arg0, %arg1, %[[C1]] : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, i64) -> !numpy.ndarray<*:!numpy.any_dtype> {sigArgTypes = ["Tensor", "Tensor", "Scalar"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
# CHECK:   %[[A1:.*]] = torch.kernel_call "aten::add" %[[A0]], %arg2, %[[C1]] : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, i64) -> !numpy.ndarray<*:!numpy.any_dtype> {sigArgTypes = ["Tensor", "Tensor", "Scalar"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
# CHECK:   return %[[A1]] : !numpy.ndarray<*:!numpy.any_dtype>
@mb.import_function
@torch.jit.script
def add3(t0, t1, t2):
  return t0 + t1 + t2

assert isinstance(add3, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
