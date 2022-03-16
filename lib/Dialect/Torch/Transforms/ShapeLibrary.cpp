//===-------------------------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file is auto-generated! Do not edit!!!
// Generated with the script `build_tools/update_shape_lib.sh`.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;

StringRef mlir::torch::Torch::getShapeLibrary() {
// TODO: Find a way to embed this string nicely.
// It is currently too long, and will probably break MSVC builds if anyone
// attempts that.
// We want to preserve the legibility of the shape library as a checked in file,
// since that is sometimes useful for debugging / diffing.
// Probably the ideal outcome is to have the shape library be a .mlir file
// that is checked in, and then we embed it as part of the build process.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
  constexpr StringLiteral shapeLib(R"mlir(
module {
  func @"__torch_mlir_shape_fn.aten.tanh"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg1: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.append.t %0, %2 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.erf"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sigmoid"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.hardsigmoid"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.square"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.hardswish"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.silu"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.hardtanh"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sqrt"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.floor"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.log2"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.rsqrt"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.abs"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.reciprocal"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.tanh_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.gelu_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.str) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.ceil"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.log"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.relu"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._softmax"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.softmax.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._log_softmax"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.log_softmax.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.clamp"(%arg0: !torch.list<int>, %arg1: !torch.optional<float>, %arg2: !torch.optional<float>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.rsub.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.to.dtype"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool, %arg3: !torch.bool, %arg4: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.to.other"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.bool, %arg4: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.type_as"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.dropout"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.gelu"(%arg0: !torch.list<int>, %arg1: !torch.str) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.contiguous"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.clone"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._log_softmax_backward_data"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.eq.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.ne.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.gt.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.ge.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.le.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.lt.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.add.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sub.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.mul.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.div.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.floor_divide.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.pow.Tensor_Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.leaky_relu"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.gather"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg2) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.layer_norm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.float, %arg5: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._softmax_backward_data"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg1) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.any"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.all"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.max"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sum"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.mean"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.var"(%arg0: !torch.list<int>, %arg1: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.std"(%arg0: !torch.list<int>, %arg1: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.argmax"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %none = torch.constant.none
    %0 = torch.aten.__is__ %arg1, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.list<int>) {
      %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %2 : !torch.list<int>
    } else {
      %2 = torch.prim.unchecked_cast %arg1 : !torch.optional<int> -> !torch.int
      %3 = call @__torch__._reduce_along_dim(%arg0, %2, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
      torch.prim.If.yield %3 : !torch.list<int>
    }
    return %1 : !torch.list<int>
  }
  func @__torch__._reduce_along_dim(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %3 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %4 = torch.prim.ListConstruct %int9223372036854775807, %3 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.prim.min.self_int %4 : !torch.list<int> -> !torch.int
    torch.prim.Loop %5, %true, init() {
    ^bb0(%arg3: !torch.int):
      %6 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
      %7 = torch.aten.eq.int %arg3, %1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %7 -> () {
        torch.prim.If %arg2 -> () {
          %8 = torch.aten.append.t %2, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %8 = torch.aten.append.t %2, %6 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %2 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.int {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.le.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.int) {
      torch.prim.If %arg2 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %2 = torch.aten.neg.int %1 : !torch.int -> !torch.int
    %3 = torch.aten.sub.int %1, %int1 : !torch.int, !torch.int -> !torch.int
    %4 = torch.aten.lt.int %arg0, %2 : !torch.int, !torch.int -> !torch.bool
    %5 = torch.prim.If %4 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %9 = torch.aten.gt.int %arg0, %3 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %9 : !torch.bool
    }
    %6 = torch.aten.__not__ %5 : !torch.bool -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.lt.int %arg0, %int0 : !torch.int, !torch.int -> !torch.bool
    %8 = torch.prim.If %7 -> (!torch.int) {
      %9 = torch.aten.add.int %arg0, %1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %9 : !torch.int
    } else {
      torch.prim.If.yield %arg0 : !torch.int
    }
    return %8 : !torch.int
  }
  func @"__torch_mlir_shape_fn.aten.any.dim"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__._reduce_along_dim(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.max.dim"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %0 = call @__torch__._reduce_along_dim(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
    %1 = torch.prim.TupleConstruct %0, %0 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %1 : !torch.tuple<list<int>, list<int>>
  }
  func @"__torch_mlir_shape_fn.aten.mean.dim"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mean_dim(%arg0, %arg1, %arg2, %0) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mean_dim(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.any) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg4: !torch.int):
      %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %3 = torch.prim.Loop %2, %true, init(%false) {
      ^bb0(%arg5: !torch.int, %arg6: !torch.bool):
        %4 = torch.aten.__getitem__.t %arg1, %arg5 : !torch.list<int>, !torch.int -> !torch.int
        %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %6 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%4, %5, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
        %7 = torch.aten.eq.int %arg4, %6 : !torch.int, !torch.int -> !torch.bool
        %8 = torch.prim.If %7 -> (!torch.bool) {
          torch.prim.If.yield %true : !torch.bool
        } else {
          torch.prim.If.yield %arg6 : !torch.bool
        }
        torch.prim.Loop.condition %true, iter(%8 : !torch.bool)
      } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
      torch.prim.If %3 -> () {
        torch.prim.If %arg2 -> () {
          %4 = torch.aten.append.t %0, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %4 = torch.aten.__getitem__.t %arg0, %arg4 : !torch.list<int>, !torch.int -> !torch.int
        %5 = torch.aten.append.t %0, %4 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sum.dim_IntList"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mean_dim(%arg0, %arg1, %arg2, %0) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.permute"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.permute(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.permute(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.eq.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %3, %true, init() {
    ^bb0(%arg2: !torch.int):
      %7 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %8 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%7, %3, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
      %9 = torch.aten.append.t %4, %8 : !torch.list<int>, !torch.int -> !torch.list<int>
      %10 = torch.aten.__getitem__.t %arg0, %8 : !torch.list<int>, !torch.int -> !torch.int
      %11 = torch.aten.append.t %5, %10 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %6 = torch.aten.__range_length %int1, %3, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %6, %true, init() {
    ^bb0(%arg2: !torch.int):
      %7 = torch.aten.__derive_index %arg2, %int1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      torch.prim.Loop %7, %true, init() {
      ^bb0(%arg3: !torch.int):
        %8 = torch.aten.__getitem__.t %4, %7 : !torch.list<int>, !torch.int -> !torch.int
        %9 = torch.aten.__getitem__.t %4, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.ne.int %8, %9 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %10 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %5 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.transpose.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.transpose(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.transpose(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg2, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = torch.aten.eq.int %1, %2 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.list<int>) {
      %5 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %0, %true, init() {
      ^bb0(%arg3: !torch.int):
        %6 = torch.aten.eq.int %arg3, %1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %6 -> () {
          %7 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
          %8 = torch.aten.append.t %5, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          %7 = torch.aten.eq.int %arg3, %2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %7 -> () {
            %8 = torch.aten.__getitem__.t %arg0, %1 : !torch.list<int>, !torch.int -> !torch.int
            %9 = torch.aten.append.t %5, %8 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.If.yield
          } else {
            %8 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
            %9 = torch.aten.append.t %5, %8 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.If.yield
          }
          torch.prim.If.yield
        }
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %5 : !torch.list<int>
    }
    return %4 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.t"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.transpose(%arg0, %int0, %int1) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.matmul"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.matmul(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.matmul(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %int-2 = torch.constant.int -2
    %true = torch.constant.bool true
    %int-1 = torch.constant.int -1
    %str = torch.constant.str "AssertionError: both  arguments to matmul need to be at least 1D"
    %none = torch.constant.none
    %0 = torch.prim.Uninitialized : !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.bool) {
      %6 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %6 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %5 = torch.prim.If %4 -> (!torch.list<int>) {
      %6 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.dot(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
      torch.prim.If.yield %6 : !torch.list<int>
    } else {
      %6 = torch.aten.eq.int %1, %int2 : !torch.int, !torch.int -> !torch.bool
      %7 = torch.prim.If %6 -> (!torch.bool) {
        %9 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %9 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %8 = torch.prim.If %7 -> (!torch.list<int>) {
        %9 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mv(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
        torch.prim.If.yield %9 : !torch.list<int>
      } else {
        %9 = torch.aten.eq.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
        %10 = torch.prim.If %9 -> (!torch.bool) {
          %12 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %12 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %11 = torch.prim.If %10 -> (!torch.list<int>) {
          %12 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unsqueeze(%arg0, %int0) : (!torch.list<int>, !torch.int) -> !torch.list<int>
          %13 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mm(%12, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
          %14 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.squeeze(%13, %int0) : (!torch.list<int>, !torch.int) -> !torch.list<int>
          torch.prim.If.yield %14 : !torch.list<int>
        } else {
          %12 = torch.aten.eq.int %1, %int2 : !torch.int, !torch.int -> !torch.bool
          %13 = torch.prim.If %12 -> (!torch.bool) {
            %15 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %15 : !torch.bool
          } else {
            torch.prim.If.yield %false : !torch.bool
          }
          %14 = torch.prim.If %13 -> (!torch.list<int>) {
            %15 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mm(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
            torch.prim.If.yield %15 : !torch.list<int>
          } else {
            %15 = torch.aten.ge.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
            %16 = torch.prim.If %15 -> (!torch.bool) {
              %18 = torch.aten.ge.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If.yield %18 : !torch.bool
            } else {
              torch.prim.If.yield %false : !torch.bool
            }
            %17 = torch.prim.If %16 -> (!torch.list<int>) {
              %18 = torch.aten.gt.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
              %19 = torch.prim.If %18 -> (!torch.int) {
                %28 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
                torch.prim.If.yield %28 : !torch.int
              } else {
                torch.prim.If.yield %int1 : !torch.int
              }
              %20 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %21 = torch.aten.sub.int %1, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %21, %true, init() {
              ^bb0(%arg2: !torch.int):
                %28 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
                %29 = torch.aten.append.t %20, %28 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %22 = torch.aten.__getitem__.t %arg1, %int-1 : !torch.list<int>, !torch.int -> !torch.int
              %23 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %24 = torch.aten.sub.int %2, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %24, %true, init() {
              ^bb0(%arg2: !torch.int):
                %28 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
                %29 = torch.aten.append.t %23, %28 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %25 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%20, %23) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
              %26 = torch.aten.gt.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %26 -> () {
                %28 = torch.aten.append.t %25, %19 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              %27 = torch.aten.gt.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %27 -> () {
                %28 = torch.aten.append.t %25, %22 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield %25 : !torch.list<int>
            } else {
              torch.prim.RaiseException %str, %none : !torch.str, !torch.none
              torch.prim.If.yield %0 : !torch.list<int>
            }
            torch.prim.If.yield %17 : !torch.list<int>
          }
          torch.prim.If.yield %14 : !torch.list<int>
        }
        torch.prim.If.yield %11 : !torch.list<int>
      }
      torch.prim.If.yield %8 : !torch.list<int>
    }
    return %5 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.dot(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %7 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %8 = torch.aten.eq.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %8 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.eq.int %3, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %6 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %6 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mv(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %8 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %9 = torch.aten.eq.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %9 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.eq.int %3, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %6 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %7 = torch.prim.ListConstruct %6 : (!torch.int) -> !torch.list<int>
    return %7 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.squeeze(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %1, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %3, %true, init() {
    ^bb0(%arg2: !torch.int):
      %4 = torch.aten.eq.int %arg2, %2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %4 -> () {
        %5 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %6 = torch.aten.ne.int %5, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %6 -> () {
          %7 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %8 = torch.aten.append.t %0, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %5 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %6 = torch.aten.append.t %0, %5 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mm(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: self must be a matrix"
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %str_1 = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %6 = torch.aten.eq.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<int>
    return %9 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unsqueeze(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.add.int %0, %int1 : !torch.int, !torch.int -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %1, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
    torch.aten.insert.t %3, %2, %int1 : !torch.list<int>, !torch.int, !torch.int
    return %3 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_0 = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.prim.max.int %0, %1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %2, %true, init() {
    ^bb0(%arg2: !torch.int):
      %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
      %5 = torch.aten.sub.int %4, %arg2 : !torch.int, !torch.int -> !torch.int
      %6 = torch.aten.sub.int %0, %int1 : !torch.int, !torch.int -> !torch.int
      %7 = torch.aten.sub.int %6, %5 : !torch.int, !torch.int -> !torch.int
      %8 = torch.aten.sub.int %1, %int1 : !torch.int, !torch.int -> !torch.int
      %9 = torch.aten.sub.int %8, %5 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.ge.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
      %11 = torch.prim.If %10 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg0, %7 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %12 = torch.aten.ge.int %9, %int0 : !torch.int, !torch.int -> !torch.bool
      %13 = torch.prim.If %12 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg1, %9 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %14 = torch.aten.ne.int %11, %13 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.bool) {
        %20 = torch.aten.ne.int %11, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %16 = torch.prim.If %15 -> (!torch.bool) {
        %20 = torch.aten.ne.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %16 -> () {
        %20 = torch.aten.format(%str, %11, %13, %arg2) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %21 = torch.aten.add.str %str_0, %20 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %21, %none : !torch.str, !torch.none
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      %17 = torch.aten.eq.int %11, %int1 : !torch.int, !torch.int -> !torch.bool
      %18 = torch.prim.If %17 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        torch.prim.If.yield %11 : !torch.int
      }
      %19 = torch.aten.append.t %3, %18 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %3 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.mm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mm(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.addmm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float, %arg4: !torch.float) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.float to !torch.any
    %1 = torch.derefine %arg4 : !torch.float to !torch.any
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.addmm(%arg0, %arg1, %arg2, %0, %1) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.any, !torch.any) -> !torch.list<int>
    return %2 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.addmm(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.any, %arg4: !torch.any) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.mm(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.bmm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %str = torch.constant.str "AssertionError: bmm only supports 3D tensors"
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: mismatching batch dimension"
    %str_1 = torch.constant.str "AssertionError: mismatching contracting dimension"
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %6 = torch.aten.eq.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.aten.eq.int %7, %8 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %9 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %10 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %11 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %12 = torch.aten.__getitem__.t %arg1, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %13 = torch.prim.ListConstruct %10, %11, %12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    return %13 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.embedding"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.embedding(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.list<int>, !torch.list<int>, !torch.int, !torch.bool, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.embedding(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.list<int>) {
      %5 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.index_select(%arg0, %int0, %arg1) : (!torch.list<int>, !torch.int, !torch.list<int>) -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %5 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg1) : (!torch.list<int>) -> !torch.list<int>
      %6 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %7 = torch.aten.append.t %5, %6 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    }
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.index_select(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.multiply_integers(%arg2) : (!torch.list<int>) -> !torch.int
    %3 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %4 = torch.aten.le.int %3, %int1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %5 = torch.aten.eq.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %9 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %10 = torch.aten.lt.int %1, %9 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %10 : !torch.bool
    }
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %8 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %8, %true, init() {
    ^bb0(%arg3: !torch.int):
      %9 = torch.aten.eq.int %1, %arg3 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %9 -> () {
        %10 = torch.aten.append.t %7, %2 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      } else {
        %10 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %11 = torch.aten.append.t %7, %10 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %7 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.multiply_integers(%arg0: !torch.list<int>) -> !torch.int {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.mul.int %arg2, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%3 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    return %1 : !torch.int
  }
  func @"__torch_mlir_shape_fn.aten.expand"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.expand(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.expand(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int-1 = torch.constant.int -1
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.aten.ge.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %5 = torch.aten.eq.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.list<int>) {
      %7 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg1) : (!torch.list<int>) -> !torch.list<int>
      torch.prim.If.yield %7 : !torch.list<int>
    } else {
      %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %3, %true, init() {
      ^bb0(%arg2: !torch.int):
        %8 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %9 = torch.aten.sub.int %8, %arg2 : !torch.int, !torch.int -> !torch.int
        %10 = torch.aten.sub.int %4, %int1 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.sub.int %10, %9 : !torch.int, !torch.int -> !torch.int
        %12 = torch.aten.ge.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
        %13 = torch.prim.If %12 -> (!torch.int) {
          %20 = torch.aten.__getitem__.t %arg0, %11 : !torch.list<int>, !torch.int -> !torch.int
          torch.prim.If.yield %20 : !torch.int
        } else {
          torch.prim.If.yield %int1 : !torch.int
        }
        %14 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %15 = torch.aten.eq.int %14, %int-1 : !torch.int, !torch.int -> !torch.bool
        %16 = torch.prim.If %15 -> (!torch.int) {
          %20 = torch.aten.ge.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %20 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          torch.prim.If.yield %13 : !torch.int
        } else {
          torch.prim.If.yield %14 : !torch.int
        }
        %17 = torch.aten.ne.int %13, %16 : !torch.int, !torch.int -> !torch.bool
        %18 = torch.prim.If %17 -> (!torch.int) {
          %20 = torch.aten.eq.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %20 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          torch.prim.If.yield %16 : !torch.int
        } else {
          torch.prim.If.yield %13 : !torch.int
        }
        %19 = torch.aten.append.t %7, %18 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %7 : !torch.list<int>
    }
    return %6 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.broadcast_to"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.expand(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.view"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.view(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.view(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.numel(%arg0) : (!torch.list<int>) -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.infer_size_impl(%arg1, %0) : (!torch.list<int>, !torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.infer_size_impl(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int-1 = torch.constant.int -1
    %str = torch.constant.str "AssertionError: only one dimension can be inferred"
    %str_0 = torch.constant.str "AssertionError: invalid shape dimensions"
    %false = torch.constant.bool false
    %str_1 = torch.constant.str "AssertionError: invalid shape"
    %0 = torch.prim.Uninitialized : !torch.int
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.derefine %none : !torch.none to !torch.optional<int>
    %3:2 = torch.prim.Loop %1, %true, init(%int1, %2) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.optional<int>):
      %9 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %10 = torch.aten.eq.int %9, %int-1 : !torch.int, !torch.int -> !torch.bool
      %11:2 = torch.prim.If %10 -> (!torch.int, !torch.optional<int>) {
        %12 = torch.aten.__isnot__ %arg4, %none : !torch.optional<int>, !torch.none -> !torch.bool
        torch.prim.If %12 -> () {
          torch.prim.RaiseException %str, %none : !torch.str, !torch.none
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        %13 = torch.derefine %arg2 : !torch.int to !torch.optional<int>
        torch.prim.If.yield %arg3, %13 : !torch.int, !torch.optional<int>
      } else {
        %12 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %13 = torch.aten.ge.int %12, %int0 : !torch.int, !torch.int -> !torch.bool
        %14 = torch.prim.If %13 -> (!torch.int) {
          %15 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %16 = torch.aten.mul.int %arg3, %15 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %16 : !torch.int
        } else {
          torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
          torch.prim.If.yield %0 : !torch.int
        }
        torch.prim.If.yield %14, %arg4 : !torch.int, !torch.optional<int>
      }
      torch.prim.Loop.condition %true, iter(%11#0, %11#1 : !torch.int, !torch.optional<int>)
    } : (!torch.int, !torch.bool, !torch.int, !torch.optional<int>) -> (!torch.int, !torch.optional<int>)
    %4 = torch.aten.eq.int %arg1, %3#0 : !torch.int, !torch.int -> !torch.bool
    %5 = torch.prim.If %4 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %9 = torch.aten.__isnot__ %3#1, %none : !torch.optional<int>, !torch.none -> !torch.bool
      %10 = torch.prim.If %9 -> (!torch.bool) {
        %12 = torch.prim.unchecked_cast %3#1 : !torch.optional<int> -> !torch.int
        %13 = torch.aten.gt.int %3#0, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %13 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %11 = torch.prim.If %10 -> (!torch.bool) {
        %12 = torch.prim.unchecked_cast %3#1 : !torch.optional<int> -> !torch.int
        %13 = torch.aten.remainder.int %arg1, %3#0 : !torch.int, !torch.int -> !torch.int
        %14 = torch.aten.eq.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %14 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %11 : !torch.bool
    }
    %6 = torch.aten.__not__ %5 : !torch.bool -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    %7 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
    %8 = torch.aten.__isnot__ %3#1, %none : !torch.optional<int>, !torch.none -> !torch.bool
    torch.prim.If %8 -> () {
      %9 = torch.prim.unchecked_cast %3#1 : !torch.optional<int> -> !torch.int
      %10 = torch.aten.floordiv.int %arg1, %3#0 : !torch.int, !torch.int -> !torch.int
      %11 = torch.aten._set_item.t %7, %9, %10 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    return %7 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.numel(%arg0: !torch.list<int>) -> !torch.int {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.mul.int %arg2, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%3 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    return %1 : !torch.int
  }
  func @"__torch_mlir_shape_fn.aten.reshape"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.view(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._unsafe_view"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.resize_"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.max_pool2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.max_pool2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.max_pool2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.list<int> {
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    %str_1 = torch.constant.str "AssertionError: max_pool2d: padding must be either be a single int, or a tuple of two ints"
    %str_2 = torch.constant.str "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
    %str_3 = torch.constant.str "AssertionError: "
    %int-4 = torch.constant.int -4
    %int-3 = torch.constant.int -3
    %int-2 = torch.constant.int -2
    %int-1 = torch.constant.int -1
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %5 = torch.aten.eq.int %4, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %7 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    %10 = torch.prim.If %9 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %10 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %11 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %14 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %15 = torch.aten.eq.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %6 : !torch.int
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int1 : !torch.int, !torch.int -> !torch.bool
      %48 = torch.prim.If %47 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        %49 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %49 : !torch.int
      }
      torch.prim.If.yield %48 : !torch.int
    }
    %17 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %19 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %20 = torch.aten.__getitem__.t %arg3, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %21 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %20 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg3, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %24 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %25 = torch.aten.eq.int %24, %int1 : !torch.int, !torch.int -> !torch.bool
    %26 = torch.prim.If %25 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %26 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %27 = torch.aten.__getitem__.t %arg4, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %28 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %29 = torch.aten.eq.int %28, %int1 : !torch.int, !torch.int -> !torch.bool
    %30 = torch.prim.If %29 -> (!torch.int) {
      torch.prim.If.yield %27 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg4, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %31 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %32 = torch.aten.eq.int %31, %int3 : !torch.int, !torch.int -> !torch.bool
    %33 = torch.prim.If %32 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %33 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %34 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %35 = torch.aten.eq.int %34, %int4 : !torch.int, !torch.int -> !torch.bool
    %36 = torch.prim.If %35 -> (!torch.int) {
      %46 = torch.aten.__getitem__.t %arg0, %int-4 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    } else {
      torch.prim.If.yield %int1 : !torch.int
    }
    %37 = torch.aten.__getitem__.t %arg0, %int-3 : !torch.list<int>, !torch.int -> !torch.int
    %38 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
    %39 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
    %40 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pooling_output_shape(%38, %3, %20, %13, %27, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %41 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pooling_output_shape(%39, %6, %23, %16, %30, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %42 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pool2d_shape_check(%arg0, %3, %6, %13, %16, %20, %23, %27, %30, %37, %38, %39, %40, %41) : (!torch.list<int>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.none
    %43 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %44 = torch.aten.eq.int %43, %int3 : !torch.int, !torch.int -> !torch.bool
    %45 = torch.prim.If %44 -> (!torch.list<int>) {
      %46 = torch.prim.ListConstruct %37, %40, %41 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %46 : !torch.list<int>
    } else {
      %46 = torch.prim.ListConstruct %36, %37, %40, %41 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %46 : !torch.list<int>
    }
    return %45 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pooling_output_shape(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.bool) -> !torch.int {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: stride should not be zeero"
    %none = torch.constant.none
    %0 = torch.aten.ne.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pooling_output_shape_pad_lr(%arg0, %arg1, %arg2, %arg2, %arg3, %arg4, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    return %1 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pooling_output_shape_pad_lr(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.bool) -> !torch.int {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.int %arg0, %arg2 : !torch.int, !torch.int -> !torch.int
    %1 = torch.aten.add.int %0, %arg3 : !torch.int, !torch.int -> !torch.int
    %2 = torch.aten.sub.int %arg1, %int1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.aten.mul.int %arg5, %2 : !torch.int, !torch.int -> !torch.int
    %4 = torch.aten.sub.int %1, %3 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.sub.int %4, %int1 : !torch.int, !torch.int -> !torch.int
    %6 = torch.prim.If %arg6 -> (!torch.int) {
      %11 = torch.aten.sub.int %arg4, %int1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %11 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %7 = torch.aten.add.int %5, %6 : !torch.int, !torch.int -> !torch.int
    %8 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.div_rtn(%7, %arg4) : (!torch.int, !torch.int) -> !torch.int
    %9 = torch.aten.add.int %8, %int1 : !torch.int, !torch.int -> !torch.int
    %10 = torch.prim.If %arg6 -> (!torch.int) {
      %11 = torch.aten.sub.int %9, %int1 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.mul.int %11, %arg4 : !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.add.int %arg0, %arg2 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.ge.int %12, %13 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.int) {
        %16 = torch.aten.sub.int %9, %int1 : !torch.int, !torch.int -> !torch.int
        torch.prim.If.yield %16 : !torch.int
      } else {
        torch.prim.If.yield %9 : !torch.int
      }
      torch.prim.If.yield %15 : !torch.int
    } else {
      torch.prim.If.yield %9 : !torch.int
    }
    return %10 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.div_rtn(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
    %0 = torch.aten.floordiv.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
    return %0 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.pool2d_shape_check(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.int, %arg7: !torch.int, %arg8: !torch.int, %arg9: !torch.int, %arg10: !torch.int, %arg11: !torch.int, %arg12: !torch.int, %arg13: !torch.int) -> !torch.none {
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.gt.int %arg2, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %19 = torch.aten.gt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %19 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.gt.int %arg4, %int0 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.bool) {
      %19 = torch.aten.gt.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %19 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %5 = torch.aten.gt.int %arg7, %int0 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      %19 = torch.aten.gt.int %arg8, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %19 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.ne.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      %19 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
      %20 = torch.aten.ne.int %19, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %20 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %10 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    %11 = torch.prim.If %10 -> (!torch.bool) {
      %19 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %20 = torch.aten.ne.int %19, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %20 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %12 = torch.prim.If %11 -> (!torch.bool) {
      torch.prim.If.yield %9 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %13 = torch.prim.If %12 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %19 = torch.aten.eq.int %0, %int4 : !torch.int, !torch.int -> !torch.bool
      %20 = torch.prim.If %19 -> (!torch.bool) {
        torch.prim.If.yield %9 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %21 = torch.prim.If %20 -> (!torch.bool) {
        %22 = torch.aten.__getitem__.t %arg0, %int3 : !torch.list<int>, !torch.int -> !torch.int
        %23 = torch.aten.ne.int %22, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %23 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %21 : !torch.bool
    }
    torch.prim.If %13 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %14 = torch.aten.floordiv.int %arg2, %int2 : !torch.int, !torch.int -> !torch.int
    %15 = torch.aten.ge.int %14, %arg6 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.bool) {
      %19 = torch.aten.floordiv.int %arg1, %int2 : !torch.int, !torch.int -> !torch.int
      %20 = torch.aten.ge.int %19, %arg5 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %20 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %16 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %17 = torch.aten.ge.int %arg13, %int1 : !torch.int, !torch.int -> !torch.bool
    %18 = torch.prim.If %17 -> (!torch.bool) {
      %19 = torch.aten.ge.int %arg12, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %19 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %18 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    return %none : !torch.none
  }
  func @"__torch_mlir_shape_fn.aten.adaptive_avg_pool2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.adaptive_avg_pool2d(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.adaptive_avg_pool2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %12 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %13 = torch.aten.eq.int %12, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %13 : !torch.bool
    }
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.__range_length %int1, %5, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %6, %true, init() {
    ^bb0(%arg2: !torch.int):
      %12 = torch.aten.__derive_index %arg2, %int1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.__getitem__.t %arg0, %12 : !torch.list<int>, !torch.int -> !torch.int
      %14 = torch.aten.ne.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %14 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %8 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %9 = torch.aten.sub.int %8, %int2 : !torch.int, !torch.int -> !torch.int
    %10 = torch.aten.__range_length %int0, %9, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %10, %true, init() {
    ^bb0(%arg2: !torch.int):
      %12 = torch.aten.__derive_index %arg2, %int0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.__getitem__.t %arg0, %12 : !torch.list<int>, !torch.int -> !torch.int
      %14 = torch.aten.append.t %7, %13 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %11 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    torch.prim.Loop %11, %true, init() {
    ^bb0(%arg2: !torch.int):
      %12 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.append.t %7, %12 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %7 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.flatten.using_ints"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.flatten(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.flatten(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg2, %2, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %4 = torch.aten.le.int %1, %3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.eq.int %5, %int0 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.list<int>) {
      %8 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %8 : !torch.list<int>
    } else {
      %8 = torch.aten.eq.int %1, %3 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.list<int>) {
        %10 = torch.prim.ListConstruct  : () -> !torch.list<int>
        %11 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        torch.prim.Loop %11, %true, init() {
        ^bb0(%arg3: !torch.int):
          %12 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
          %13 = torch.aten.append.t %10, %12 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %10 : !torch.list<int>
      } else {
        %10 = torch.aten.add.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.__range_length %1, %10, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        %12 = torch.prim.Loop %11, %true, init(%int1) {
        ^bb0(%arg3: !torch.int, %arg4: !torch.int):
          %18 = torch.aten.__derive_index %arg3, %1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %19 = torch.aten.__getitem__.t %arg0, %18 : !torch.list<int>, !torch.int -> !torch.int
          %20 = torch.aten.mul.int %arg4, %19 : !torch.int, !torch.int -> !torch.int
          torch.prim.Loop.condition %true, iter(%20 : !torch.int)
        } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
        %13 = torch.prim.ListConstruct  : () -> !torch.list<int>
        torch.prim.Loop %1, %true, init() {
        ^bb0(%arg3: !torch.int):
          %18 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
          %19 = torch.aten.append.t %13, %18 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        %14 = torch.aten.append.t %13, %12 : !torch.list<int>, !torch.int -> !torch.list<int>
        %15 = torch.aten.add.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %16 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %17 = torch.aten.__range_length %15, %16, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        torch.prim.Loop %17, %true, init() {
        ^bb0(%arg3: !torch.int):
          %18 = torch.aten.__derive_index %arg3, %15, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %19 = torch.aten.__getitem__.t %arg0, %18 : !torch.list<int>, !torch.int -> !torch.int
          %20 = torch.aten.append.t %13, %19 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %13 : !torch.list<int>
      }
      torch.prim.If.yield %9 : !torch.list<int>
    }
    return %7 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.linear"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.linear(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.linear(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.t(%arg1) : (!torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.matmul(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %2 = torch.aten.__isnot__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    torch.prim.If %2 -> () {
      %3 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%3, %1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
      %5 = torch.aten.eq.int_list %4, %1 : !torch.list<int>, !torch.list<int> -> !torch.bool
      torch.prim.If %5 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    return %1 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.t(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.le.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.list<int>) {
      %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %5 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      %6 = torch.prim.If %5 -> (!torch.list<int>) {
        %7 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %8 = torch.prim.ListConstruct %7 : (!torch.int) -> !torch.list<int>
        torch.prim.If.yield %8 : !torch.list<int>
      } else {
        %7 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
        %8 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<int>
        torch.prim.If.yield %9 : !torch.list<int>
      }
      torch.prim.If.yield %6 : !torch.list<int>
    }
    return %4 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.zeros"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.ones"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.empty.memory_format"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.full"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.full_like"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>, %arg6: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.zeros_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.ones_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.empty_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.new_zeros"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.new_ones"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.fill.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.uniform"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float, %arg3: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.bernoulli.float"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.bernoulli.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.bernoulli"(%arg0: !torch.list<int>, %arg1: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.arange.start_step"(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.float, %arg3: !torch.optional<int>, %arg4: !torch.optional<int>, %arg5: !torch.optional<Device>, %arg6: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %1 = torch.derefine %arg4 : !torch.optional<int> to !torch.any
    %2 = torch.derefine %arg5 : !torch.optional<Device> to !torch.any
    %3 = torch.derefine %arg6 : !torch.optional<bool> to !torch.any
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_start_step(%arg0, %arg1, %arg2, %0, %1, %2, %3) : (!torch.float, !torch.float, !torch.float, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_start_step(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.float, %arg3: !torch.any, %arg4: !torch.any, %arg5: !torch.any, %arg6: !torch.any) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.operator "aten.ne.float_int"(%arg2, %int0) : (!torch.float, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.aten.lt.float_int %arg2, %int0 : !torch.float, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      %6 = torch.operator "aten.ge.float"(%arg0, %arg1) : (!torch.float, !torch.float) -> !torch.bool
      torch.prim.If %6 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    } else {
      %6 = torch.operator "aten.ge.float"(%arg1, %arg0) : (!torch.float, !torch.float) -> !torch.bool
      torch.prim.If %6 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    }
    %2 = torch.operator "aten.sub.float"(%arg1, %arg0) : (!torch.float, !torch.float) -> !torch.float
    %3 = torch.operator "aten.div.float"(%2, %arg2) : (!torch.float, !torch.float) -> !torch.float
    %4 = torch.operator "aten.ceil.float"(%3) : (!torch.float) -> !torch.int
    %5 = torch.prim.ListConstruct %4 : (!torch.int) -> !torch.list<int>
    return %5 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.arange.start"(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg2 : !torch.optional<int> to !torch.any
    %1 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %2 = torch.derefine %arg4 : !torch.optional<Device> to !torch.any
    %3 = torch.derefine %arg5 : !torch.optional<bool> to !torch.any
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_start(%arg0, %arg1, %0, %1, %2, %3) : (!torch.float, !torch.float, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_start(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.any, %arg3: !torch.any, %arg4: !torch.any, %arg5: !torch.any) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.operator "aten.ge.float_int"(%arg1, %int0) : (!torch.float, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.operator "aten.ge.float"(%arg1, %arg0) : (!torch.float, !torch.float) -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.operator "aten.sub.float"(%arg1, %arg0) : (!torch.float, !torch.float) -> !torch.float
    %3 = torch.operator "aten.ceil.float"(%2) : (!torch.float) -> !torch.int
    %4 = torch.prim.ListConstruct %3 : (!torch.int) -> !torch.list<int>
    return %4 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.arange"(%arg0: !torch.float, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg1 : !torch.optional<int> to !torch.any
    %1 = torch.derefine %arg2 : !torch.optional<int> to !torch.any
    %2 = torch.derefine %arg3 : !torch.optional<Device> to !torch.any
    %3 = torch.derefine %arg4 : !torch.optional<bool> to !torch.any
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_end(%arg0, %0, %1, %2, %3) : (!torch.float, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.arange_end(%arg0: !torch.float, %arg1: !torch.any, %arg2: !torch.any, %arg3: !torch.any, %arg4: !torch.any) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.operator "aten.ge.float_int"(%arg0, %int0) : (!torch.float, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.operator "aten.ceil.float"(%arg0) : (!torch.float) -> !torch.int
    %2 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
    return %2 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.add.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.sub.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.mul.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.div.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.__and__.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.minimum"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.maximum"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.bitwise_and.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.threshold"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.threshold_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.eq.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.gt.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.lt.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.unsqueeze"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unsqueeze(%arg0, %arg1) : (!torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.squeeze"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.squeeze_nodim(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.squeeze_nodim(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg1: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.ne.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %3 -> () {
        %4 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
        %5 = torch.aten.append.t %0, %4 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.squeeze.dim"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.squeeze(%arg0, %arg1) : (!torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.prim.NumToTensor.Scalar"(%arg0: !torch.float) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.tensor.float"(%arg0: !torch.float, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.tensor.int"(%arg0: !torch.int, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.tensor.bool"(%arg0: !torch.bool, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten._shape_as_tensor"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.where.self"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.lerp.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.addcmul"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.addcdiv"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.topk"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %str = torch.constant.str "k ({}) is too big for dimension {} of size {}"
    %str_0 = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
    %1 = torch.aten.le.int %arg1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      %4 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.aten.format(%str, %arg1, %arg2, %4) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
      %6 = torch.aten.add.str %str_0, %5 : !torch.str, !torch.str -> !torch.str
      torch.prim.RaiseException %6, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten._set_item.t %arg0, %arg2, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    %3 = torch.prim.TupleConstruct %arg0, %arg0 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %3 : !torch.tuple<list<int>, list<int>>
  }
  func @"__torch_mlir_shape_fn.aten.conv2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.conv2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.conv2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %int4 = torch.constant.int 4
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.conv_output_size(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.list<int>
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.conv_output_size(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_shape_forward(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.none
    %1 = torch.aten.len.t %arg5 : !torch.list<int> -> !torch.int
    %2 = torch.aten.gt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %5 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %6 = torch.aten.append.t %4, %5 : !torch.list<int>, !torch.int -> !torch.list<int>
    %7 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.append.t %4, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
    %9 = torch.aten.__range_length %int2, %3, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %9, %true, init() {
    ^bb0(%arg7: !torch.int):
      %10 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %11 = torch.prim.If %2 -> (!torch.int) {
        %27 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
        %28 = torch.aten.__getitem__.t %arg5, %27 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %12 = torch.aten.__getitem__.t %arg1, %10 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.sub.int %12, %int1 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.mul.int %11, %13 : !torch.int, !torch.int -> !torch.int
      %15 = torch.aten.add.int %14, %int1 : !torch.int, !torch.int -> !torch.int
      %16 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %17 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
      %18 = torch.aten.__getitem__.t %arg4, %17 : !torch.list<int>, !torch.int -> !torch.int
      %19 = torch.aten.mul.int %int2, %18 : !torch.int, !torch.int -> !torch.int
      %20 = torch.aten.add.int %16, %19 : !torch.int, !torch.int -> !torch.int
      %21 = torch.aten.sub.int %20, %15 : !torch.int, !torch.int -> !torch.int
      %22 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
      %23 = torch.aten.__getitem__.t %arg3, %22 : !torch.list<int>, !torch.int -> !torch.int
      %24 = torch.aten.floordiv.int %21, %23 : !torch.int, !torch.int -> !torch.int
      %25 = torch.aten.add.int %24, %int1 : !torch.int, !torch.int -> !torch.int
      %26 = torch.aten.append.t %4, %25 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %4 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_shape_forward(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.none {
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_non_negative(%arg4) : (!torch.list<int>) -> !torch.bool
    %3 = torch.aten.__not__ %2 : !torch.bool -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_non_negative(%arg3) : (!torch.list<int>) -> !torch.bool
    %5 = torch.aten.__not__ %4 : !torch.bool -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %6 = torch.aten.eq.int %1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.ge.int %7, %arg6 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %10 = torch.aten.remainder.int %9, %arg6 : !torch.int, !torch.int -> !torch.int
    %11 = torch.aten.eq.int %10, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %13 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %14 = torch.aten.mul.int %13, %arg6 : !torch.int, !torch.int -> !torch.int
    %15 = torch.aten.eq.int %12, %14 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %15 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %16 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %17 = torch.prim.If %16 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %19 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %20 = torch.aten.len.t %19 : !torch.list<int> -> !torch.int
      %21 = torch.aten.eq.int %20, %int1 : !torch.int, !torch.int -> !torch.bool
      %22 = torch.prim.If %21 -> (!torch.bool) {
        %23 = torch.aten.__getitem__.t %19, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %24 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %25 = torch.aten.eq.int %23, %24 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %25 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %22 : !torch.bool
    }
    torch.prim.If %17 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %18 = torch.aten.__range_length %int2, %0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %18, %true, init() {
    ^bb0(%arg7: !torch.int):
      %19 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %20 = torch.aten.__getitem__.t %arg0, %19 : !torch.list<int>, !torch.int -> !torch.int
      %21 = torch.aten.sub.int %19, %int2 : !torch.int, !torch.int -> !torch.int
      %22 = torch.aten.__getitem__.t %arg4, %21 : !torch.list<int>, !torch.int -> !torch.int
      %23 = torch.aten.mul.int %int2, %22 : !torch.int, !torch.int -> !torch.int
      %24 = torch.aten.add.int %20, %23 : !torch.int, !torch.int -> !torch.int
      %25 = torch.aten.sub.int %19, %int2 : !torch.int, !torch.int -> !torch.int
      %26 = torch.aten.__getitem__.t %arg5, %25 : !torch.list<int>, !torch.int -> !torch.int
      %27 = torch.aten.__getitem__.t %arg1, %19 : !torch.list<int>, !torch.int -> !torch.int
      %28 = torch.aten.sub.int %27, %int1 : !torch.int, !torch.int -> !torch.int
      %29 = torch.aten.mul.int %26, %28 : !torch.int, !torch.int -> !torch.int
      %30 = torch.aten.add.int %29, %int1 : !torch.int, !torch.int -> !torch.int
      %31 = torch.aten.ge.int %24, %30 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %31 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %none : !torch.none
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_non_negative(%arg0: !torch.list<int>) -> !torch.bool {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%false) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.bool):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.lt.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
      %4 = torch.prim.If %3 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg2 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%4 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    return %1 : !torch.bool
  }
  func @"__torch_mlir_shape_fn.aten.batch_norm"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.slice.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.slice(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.list<int>, !torch.int, !torch.optional<int>, !torch.optional<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.slice(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.ne.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = torch.aten.__isnot__ %arg2, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.int) {
      %25 = torch.prim.unchecked_cast %arg2 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %25 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %5 = torch.aten.__isnot__ %arg3, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      %25 = torch.prim.unchecked_cast %arg3 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %25 : !torch.int
    } else {
      %25 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.max_int() : () -> !torch.int
      torch.prim.If.yield %25 : !torch.int
    }
    %7 = torch.aten.gt.int %arg4, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.max_int() : () -> !torch.int
    %9 = torch.aten.eq.int %4, %8 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.int) {
      torch.prim.If.yield %int0 : !torch.int
    } else {
      torch.prim.If.yield %4 : !torch.int
    }
    %11 = torch.aten.lt.int %10, %int0 : !torch.int, !torch.int -> !torch.bool
    %12 = torch.prim.If %11 -> (!torch.int) {
      %25 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
      %26 = torch.aten.add.int %10, %25 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %26 : !torch.int
    } else {
      torch.prim.If.yield %10 : !torch.int
    }
    %13 = torch.aten.lt.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
    %14 = torch.prim.If %13 -> (!torch.int) {
      %25 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
      %26 = torch.aten.add.int %6, %25 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %26 : !torch.int
    } else {
      torch.prim.If.yield %6 : !torch.int
    }
    %15 = torch.aten.lt.int %12, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %int0 : !torch.int
    } else {
      %25 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
      %26 = torch.aten.ge.int %12, %25 : !torch.int, !torch.int -> !torch.bool
      %27 = torch.prim.If %26 -> (!torch.int) {
        %28 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %12 : !torch.int
      }
      torch.prim.If.yield %27 : !torch.int
    }
    %17 = torch.aten.lt.int %14, %16 : !torch.int, !torch.int -> !torch.bool
    %18 = torch.prim.If %17 -> (!torch.int) {
      torch.prim.If.yield %16 : !torch.int
    } else {
      %25 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
      %26 = torch.aten.ge.int %14, %25 : !torch.int, !torch.int -> !torch.bool
      %27 = torch.prim.If %26 -> (!torch.int) {
        %28 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %14 : !torch.int
      }
      torch.prim.If.yield %27 : !torch.int
    }
    %19 = torch.aten.sub.int %18, %16 : !torch.int, !torch.int -> !torch.int
    %20 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
    %21 = torch.aten.add.int %19, %arg4 : !torch.int, !torch.int -> !torch.int
    %22 = torch.aten.sub.int %21, %int1 : !torch.int, !torch.int -> !torch.int
    %23 = torch.aten.floordiv.int %22, %arg4 : !torch.int, !torch.int -> !torch.int
    %24 = torch.aten._set_item.t %20, %2, %23 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    return %20 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.max_int() -> !torch.int {
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    return %int9223372036854775807 : !torch.int
  }
  func @"__torch_mlir_shape_fn.aten.select.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.select(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.select(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.ne.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = torch.aten.__getitem__.t %arg0, %2 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.neg.int %3 : !torch.int -> !torch.int
    %5 = torch.aten.lt.int %arg2, %4 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %9 = torch.aten.ge.int %arg2, %3 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %9 : !torch.bool
    }
    %7 = torch.aten.__not__ %6 : !torch.bool -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %0, %true, init() {
    ^bb0(%arg3: !torch.int):
      %9 = torch.aten.ne.int %arg3, %2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %9 -> () {
        %10 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %11 = torch.aten.append.t %8, %10 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %8 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.index_select"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.index_select(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.nll_loss_forward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.int, %arg4: !torch.int) -> !torch.tuple<list<int>, list<int>> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int-1 = torch.constant.int -1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.lt.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.bool) {
      %15 = torch.aten.le.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %15 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.le.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %5 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      %15 = torch.aten.eq.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %15 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %15 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %16 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %17 = torch.aten.eq.int %15, %16 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %17 : !torch.bool
    }
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %10 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %11 = torch.prim.If %10 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %15 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %16 = torch.aten.len.t %15 : !torch.list<int> -> !torch.int
      %17 = torch.aten.eq.int %16, %int1 : !torch.int, !torch.int -> !torch.bool
      %18 = torch.prim.If %17 -> (!torch.bool) {
        %19 = torch.aten.__getitem__.t %15, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.eq.int %19, %8 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %18 : !torch.bool
    }
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.eq.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.bool) {
      %15 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %15 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %14 = torch.prim.If %13 -> (!torch.tuple<list<int>, list<int>>) {
      %15 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %16 = torch.prim.ListConstruct %15 : (!torch.int) -> !torch.list<int>
      %17 = torch.prim.TupleConstruct %16, %9 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
      torch.prim.If.yield %17 : !torch.tuple<list<int>, list<int>>
    } else {
      %15 = torch.prim.TupleConstruct %9, %9 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
      torch.prim.If.yield %15 : !torch.tuple<list<int>, list<int>>
    }
    return %14 : !torch.tuple<list<int>, list<int>>
  }
  func @"__torch_mlir_shape_fn.aten.nll_loss_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.optional<list<int>>, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.unary(%arg1) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.native_layer_norm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.float) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.sub.int %1, %2 : !torch.int, !torch.int -> !torch.int
    %4 = torch.aten.ge.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    torch.prim.Loop %3, %true, init() {
    ^bb0(%arg5: !torch.int):
      %8 = torch.aten.__getitem__.t %arg0, %arg5 : !torch.list<int>, !torch.int -> !torch.int
      %9 = torch.aten.append.t %0, %8 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.__range_length %3, %5, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %6, %true, init() {
    ^bb0(%arg5: !torch.int):
      %8 = torch.aten.append.t %0, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %7 = torch.prim.TupleConstruct %arg0, %0, %0 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
    return %7 : !torch.tuple<list<int>, list<int>, list<int>>
  }
  func @"__torch_mlir_shape_fn.aten.native_batch_norm"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.If %arg5 -> (!torch.tuple<list<int>, list<int>, list<int>>) {
      %1 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %2 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
      %3 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %4 = torch.prim.ListConstruct %3 : (!torch.int) -> !torch.list<int>
      %5 = torch.prim.TupleConstruct %arg0, %2, %4 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
      torch.prim.If.yield %5 : !torch.tuple<list<int>, list<int>, list<int>>
    } else {
      %1 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
      %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
      %3 = torch.prim.TupleConstruct %arg0, %1, %2 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
      torch.prim.If.yield %3 : !torch.tuple<list<int>, list<int>, list<int>>
    }
    return %0 : !torch.tuple<list<int>, list<int>, list<int>>
  }
  func @"__torch_mlir_shape_fn.aten.constant_pad_nd"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: Must have paired low-high pad amount values"
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: Number of padded dimensions must be less than or equal to the input dimension"
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.remainder.int %0, %int2 : !torch.int, !torch.int -> !torch.int
    %2 = torch.aten.eq.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %4 = torch.aten.floordiv.int %3, %int2 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.le.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %8 = torch.aten.floordiv.int %7, %int2 : !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %8, %true, init() {
    ^bb0(%arg3: !torch.int):
      %9 = torch.aten.add.int %arg3, %int1 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.neg.int %9 : !torch.int -> !torch.int
      %11 = torch.aten.mul.int %int2, %arg3 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.__getitem__.t %arg1, %11 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.mul.int %int2, %arg3 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.add.int %13, %int1 : !torch.int, !torch.int -> !torch.int
      %15 = torch.aten.__getitem__.t %arg1, %14 : !torch.list<int>, !torch.int -> !torch.int
      %16 = torch.aten.add.int %12, %15 : !torch.int, !torch.int -> !torch.int
      %17 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %18 = torch.aten.add.int %17, %16 : !torch.int, !torch.int -> !torch.int
      %19 = torch.aten._set_item.t %arg0, %10, %18 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %arg0 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.index.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<optional<list<int>>>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: More indices than dimensions to index"
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg1 : !torch.list<optional<list<int>>> -> !torch.int
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.aten.le.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %4 = torch.aten.len.t %arg1 : !torch.list<optional<list<int>>> -> !torch.int
    %5 = torch.prim.Loop %4, %true, init(%3) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.list<int>):
      %6 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<optional<list<int>>>, !torch.int -> !torch.optional<list<int>>
      %7 = torch.aten.__isnot__ %6, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
      %8 = torch.prim.If %7 -> (!torch.list<int>) {
        %9 = torch.prim.unchecked_cast %6 : !torch.optional<list<int>> -> !torch.list<int>
        %10 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.broadcast(%arg3, %9) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
        torch.prim.If.yield %10 : !torch.list<int>
      } else {
        torch.prim.If.yield %arg3 : !torch.list<int>
      }
      torch.prim.Loop.condition %true, iter(%8 : !torch.list<int>)
    } : (!torch.int, !torch.bool, !torch.list<int>) -> !torch.list<int>
    return %5 : !torch.list<int>
  }
  func @"__torch_mlir_shape_fn.aten.cat"(%arg0: !torch.list<list<int>>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.cat(%arg0, %arg1) : (!torch.list<list<int>>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.cat(%arg0: !torch.list<list<int>>, %arg1: !torch.int) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_cat_no_zero_dim(%arg0) : (!torch.list<list<int>>) -> !torch.none
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.legacy_cat_wrap_dim(%arg1, %arg0) : (!torch.int, !torch.list<list<int>>) -> !torch.int
    %2 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    %3 = torch.aten.gt.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    %5 = torch.derefine %none : !torch.none to !torch.optional<list<int>>
    %6 = torch.prim.Loop %4, %true, init(%5) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.optional<list<int>>):
      %9 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %10 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.should_skip(%9) : (!torch.list<int>) -> !torch.bool
      %11 = torch.aten.__not__ %10 : !torch.bool -> !torch.bool
      %12 = torch.prim.If %11 -> (!torch.optional<list<int>>) {
        %13 = torch.derefine %9 : !torch.list<int> to !torch.optional<list<int>>
        torch.prim.If.yield %13 : !torch.optional<list<int>>
      } else {
        torch.prim.If.yield %arg3 : !torch.optional<list<int>>
      }
      torch.prim.Loop.condition %true, iter(%12 : !torch.optional<list<int>>)
    } : (!torch.int, !torch.bool, !torch.optional<list<int>>) -> !torch.optional<list<int>>
    %7 = torch.aten.__is__ %6, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %8 = torch.prim.If %7 -> (!torch.list<int>) {
      %9 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %9 : !torch.list<int>
    } else {
      %9 = torch.prim.unchecked_cast %6 : !torch.optional<list<int>> -> !torch.list<int>
      %10 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
      %11 = torch.prim.Loop %10, %true, init(%int0) {
      ^bb0(%arg2: !torch.int, %arg3: !torch.int):
        %14 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
        %15 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.should_skip(%14) : (!torch.list<int>) -> !torch.bool
        %16 = torch.aten.__not__ %15 : !torch.bool -> !torch.bool
        %17 = torch.prim.If %16 -> (!torch.int) {
          %18 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_cat_shape_except_dim(%9, %14, %1, %arg2) : (!torch.list<int>, !torch.list<int>, !torch.int, !torch.int) -> !torch.none
          %19 = torch.aten.__getitem__.t %14, %1 : !torch.list<int>, !torch.int -> !torch.int
          %20 = torch.aten.add.int %arg3, %19 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %20 : !torch.int
        } else {
          torch.prim.If.yield %arg3 : !torch.int
        }
        torch.prim.Loop.condition %true, iter(%17 : !torch.int)
      } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
      %12 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers._copy(%9) : (!torch.list<int>) -> !torch.list<int>
      %13 = torch.aten._set_item.t %12, %1, %11 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield %12 : !torch.list<int>
    }
    return %8 : !torch.list<int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_cat_no_zero_dim(%arg0: !torch.list<list<int>>) -> !torch.none {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    torch.prim.Loop %0, %true, init() {
    ^bb0(%arg1: !torch.int):
      %1 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %2 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int
      %3 = torch.aten.gt.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %3 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %none : !torch.none
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.legacy_cat_wrap_dim(%arg0: !torch.int, %arg1: !torch.list<list<int>>) -> !torch.int {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %0 = torch.aten.len.t %arg1 : !torch.list<list<int>> -> !torch.int
    %1 = torch.derefine %none : !torch.none to !torch.optional<int>
    %2 = torch.prim.Loop %0, %true, init(%1) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.optional<int>):
      %5 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %6 = torch.aten.len.t %5 : !torch.list<int> -> !torch.int
      %7 = torch.aten.ne.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
      %8 = torch.prim.If %7 -> (!torch.bool) {
        %11 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
        %12 = torch.aten.ne.int_list %5, %11 : !torch.list<int>, !torch.list<int> -> !torch.bool
        torch.prim.If.yield %12 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %9 = torch.prim.If %8 -> (!torch.bool) {
        %11 = torch.aten.__isnot__ %arg3, %none : !torch.optional<int>, !torch.none -> !torch.bool
        torch.prim.If.yield %11 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %10 = torch.prim.If %9 -> (!torch.optional<int>) {
        %11 = torch.aten.len.t %5 : !torch.list<int> -> !torch.int
        %12 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.maybe_wrap_dim(%arg0, %11, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
        %13 = torch.derefine %12 : !torch.int to !torch.optional<int>
        torch.prim.If.yield %13 : !torch.optional<int>
      } else {
        torch.prim.If.yield %arg3 : !torch.optional<int>
      }
      torch.prim.Loop.condition %true, iter(%10 : !torch.optional<int>)
    } : (!torch.int, !torch.bool, !torch.optional<int>) -> !torch.optional<int>
    %3 = torch.aten.__is__ %2, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.int) {
      torch.prim.If.yield %arg0 : !torch.int
    } else {
      %5 = torch.prim.unchecked_cast %2 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %5 : !torch.int
    }
    return %4 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.should_skip(%arg0: !torch.list<int>) -> !torch.bool {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.numel(%arg0) : (!torch.list<int>) -> !torch.int
    %1 = torch.aten.eq.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %3 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %4 = torch.aten.eq.int %3, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %4 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    return %2 : !torch.bool
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.upstream_shape_helpers.check_cat_shape_except_dim(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.none {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: Tensors must have same number of dimensions"
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %str_0 = torch.constant.str "AssertionError: Sizes of tensors must match except in dimension"
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.eq.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__range_length %int0, %0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %3, %true, init() {
    ^bb0(%arg4: !torch.int):
      %4 = torch.aten.__derive_index %arg4, %int0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %5 = torch.aten.ne.int %4, %arg2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %5 -> () {
        %6 = torch.aten.__getitem__.t %arg0, %4 : !torch.list<int>, !torch.int -> !torch.int
        %7 = torch.aten.__getitem__.t %arg1, %4 : !torch.list<int>, !torch.int -> !torch.int
        %8 = torch.aten.eq.int %6, %7 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %8 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %none : !torch.none
  }
  func @"__torch_mlir_shape_fn.aten.bincount"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.hacky_get_unknown_dimension_size() : () -> !torch.int
    %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func @__torch__.hacky_get_unknown_dimension_size() -> !torch.int {
    %0 = torch.prim.CreateObject !torch.nn.Module<"__torch__.DummyClassType">
    %1 = torch.prim.CallMethod %0["__init__"] () : !torch.nn.Module<"__torch__.DummyClassType">, () -> !torch.none
    %2 = torch.operator "prim.id"(%0) : (!torch.nn.Module<"__torch__.DummyClassType">) -> !torch.int
    return %2 : !torch.int
  }
  func @__torch__.DummyClassType.__init__(%arg0: !torch.nn.Module<"__torch__.DummyClassType">) -> !torch.none {
    %none = torch.constant.none
    return %none : !torch.none
  }
}
)mlir");
#pragma clang diagnostic pop
  return shapeLib;
}
