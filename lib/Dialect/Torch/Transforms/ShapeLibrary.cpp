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
  constexpr StringLiteral shapeLib(R"mlir(
module  {
  func @"__torch_mlir_shape_fn.aten.tanh"(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unary(%arg0) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unary(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers._copy(%arg0) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers._copy(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    %1 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    torch.prim.Loop %1, %true, init()  {
    ^bb0(%arg1: !torch.int):  // no predecessors
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %3 = torch.aten.append.t %0, %2 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.relu"(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unary(%arg0) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.max_pool2d"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.list<!torch.int>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.bool) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.max_pool2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.bool) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.max_pool2d(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.list<!torch.int>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.bool) -> !torch.list<!torch.int> {
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    %str_0 = torch.constant.str "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    %str_1 = torch.constant.str "AssertionError: max_pool2d: padding must be either be a single int, or a tuple of two ints"
    %str_2 = torch.constant.str "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
    %str_3 = torch.constant.str "AssertionError: "
    %int-4 = torch.constant.int -4
    %int-3 = torch.constant.int -3
    %int-2 = torch.constant.int -2
    %int-1 = torch.constant.int -1
    %0 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %4 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %5 = torch.aten.eq.int %4, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %7 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    %10 = torch.prim.If %9 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %10 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0 : !torch.str
      torch.prim.If.yield
    }
    %11 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
    %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %14 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
    %15 = torch.aten.eq.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %6 : !torch.int
    } else {
      %46 = torch.aten.len.t %arg2 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int1 : !torch.int, !torch.int -> !torch.bool
      %48 = torch.prim.If %47 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        %49 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
        torch.prim.If.yield %49 : !torch.int
      }
      torch.prim.If.yield %48 : !torch.int
    }
    %17 = torch.aten.len.t %arg3 : !torch.list<!torch.int> -> !torch.int
    %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg3 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %19 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1 : !torch.str
      torch.prim.If.yield
    }
    %20 = torch.aten.__getitem__.t %arg3, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %21 = torch.aten.len.t %arg3 : !torch.list<!torch.int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %20 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg3, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %24 = torch.aten.len.t %arg4 : !torch.list<!torch.int> -> !torch.int
    %25 = torch.aten.eq.int %24, %int1 : !torch.int, !torch.int -> !torch.bool
    %26 = torch.prim.If %25 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg4 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %26 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2 : !torch.str
      torch.prim.If.yield
    }
    %27 = torch.aten.__getitem__.t %arg4, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %28 = torch.aten.len.t %arg4 : !torch.list<!torch.int> -> !torch.int
    %29 = torch.aten.eq.int %28, %int1 : !torch.int, !torch.int -> !torch.bool
    %30 = torch.prim.If %29 -> (!torch.int) {
      torch.prim.If.yield %27 : !torch.int
    } else {
      %46 = torch.aten.__getitem__.t %arg4, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    }
    %31 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %32 = torch.aten.eq.int %31, %int3 : !torch.int, !torch.int -> !torch.bool
    %33 = torch.prim.If %32 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %46 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
      %47 = torch.aten.eq.int %46, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %47 : !torch.bool
    }
    torch.prim.If %33 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_3 : !torch.str
      torch.prim.If.yield
    }
    %34 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %35 = torch.aten.eq.int %34, %int4 : !torch.int, !torch.int -> !torch.bool
    %36 = torch.prim.If %35 -> (!torch.int) {
      %46 = torch.aten.__getitem__.t %arg0, %int-4 : !torch.list<!torch.int>, !torch.int -> !torch.int
      torch.prim.If.yield %46 : !torch.int
    } else {
      torch.prim.If.yield %int1 : !torch.int
    }
    %37 = torch.aten.__getitem__.t %arg0, %int-3 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %38 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %39 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %40 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pooling_output_shape(%38, %3, %20, %13, %27, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %41 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pooling_output_shape(%39, %6, %23, %16, %30, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %42 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pool2d_shape_check(%arg0, %3, %6, %13, %16, %20, %23, %27, %30, %37, %38, %39, %40, %41) : (!torch.list<!torch.int>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.none
    %43 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %44 = torch.aten.eq.int %43, %int3 : !torch.int, !torch.int -> !torch.bool
    %45 = torch.prim.If %44 -> (!torch.list<!torch.int>) {
      %46 = torch.prim.ListConstruct %37, %40, %41 : (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
      torch.prim.If.yield %46 : !torch.list<!torch.int>
    } else {
      %46 = torch.prim.ListConstruct %36, %37, %40, %41 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
      torch.prim.If.yield %46 : !torch.list<!torch.int>
    }
    return %45 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pooling_output_shape(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.bool) -> !torch.int {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: stride should not be zeero"
    %0 = torch.aten.ne.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pooling_output_shape_pad_lr(%arg0, %arg1, %arg2, %arg2, %arg3, %arg4, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    return %1 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pooling_output_shape_pad_lr(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.bool) -> !torch.int {
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
    %8 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.div_rtn(%7, %arg4) : (!torch.int, !torch.int) -> !torch.int
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
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.div_rtn(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
    %0 = torch.aten.floordiv.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
    return %0 : !torch.int
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.pool2d_shape_check(%arg0: !torch.list<!torch.int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.int, %arg7: !torch.int, %arg8: !torch.int, %arg9: !torch.int, %arg10: !torch.int, %arg11: !torch.int, %arg12: !torch.int, %arg13: !torch.int) -> !torch.none {
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
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
      torch.prim.RaiseException %str : !torch.str
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
      torch.prim.RaiseException %str : !torch.str
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
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %8 = torch.aten.ne.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      %19 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %20 = torch.aten.ne.int %19, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %20 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %10 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    %11 = torch.prim.If %10 -> (!torch.bool) {
      %19 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
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
        %22 = torch.aten.__getitem__.t %arg0, %int3 : !torch.list<!torch.int>, !torch.int -> !torch.int
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
      torch.prim.RaiseException %str : !torch.str
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
      torch.prim.RaiseException %str : !torch.str
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
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    return %none : !torch.none
  }
  func @"__torch_mlir_shape_fn.aten.adaptive_avg_pool2d"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.adaptive_avg_pool2d(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.adaptive_avg_pool2d(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %12 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
      %13 = torch.aten.eq.int %12, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %13 : !torch.bool
    }
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %5 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %6 = torch.aten.__range_length %int1, %5, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %6, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %12 = torch.aten.__derive_index %arg2, %int1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.__getitem__.t %arg0, %12 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %14 = torch.aten.ne.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %14 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str : !torch.str
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %7 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    %8 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %9 = torch.aten.sub.int %8, %int2 : !torch.int, !torch.int -> !torch.int
    %10 = torch.aten.__range_length %int0, %9, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %10, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %12 = torch.aten.__derive_index %arg2, %int0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.__getitem__.t %arg0, %12 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %14 = torch.aten.append.t %7, %13 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %11 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    torch.prim.Loop %11, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %12 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %13 = torch.aten.append.t %7, %12 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %7 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.flatten.using_ints"(%arg0: !torch.list<!torch.int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.flatten(%arg0, %arg1, %arg2) : (!torch.list<!torch.int>, !torch.int, !torch.int) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.flatten(%arg0: !torch.list<!torch.int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<!torch.int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %2 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %3 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.maybe_wrap_dim(%arg2, %2, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %4 = torch.aten.le.int %1, %3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %4 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %5 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %6 = torch.aten.eq.int %5, %int0 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.list<!torch.int>) {
      %8 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<!torch.int>
      torch.prim.If.yield %8 : !torch.list<!torch.int>
    } else {
      %8 = torch.aten.eq.int %1, %3 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.list<!torch.int>) {
        %10 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
        %11 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
        torch.prim.Loop %11, %true, init()  {
        ^bb0(%arg3: !torch.int):  // no predecessors
          %12 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<!torch.int>, !torch.int -> !torch.int
          %13 = torch.aten.append.t %10, %12 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %10 : !torch.list<!torch.int>
      } else {
        %10 = torch.aten.add.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.__range_length %1, %10, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        %12 = torch.prim.Loop %11, %true, init(%int1)  {
        ^bb0(%arg3: !torch.int, %arg4: !torch.int):  // no predecessors
          %18 = torch.aten.__derive_index %arg3, %1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %19 = torch.aten.__getitem__.t %arg0, %18 : !torch.list<!torch.int>, !torch.int -> !torch.int
          %20 = torch.aten.mul.int %arg4, %19 : !torch.int, !torch.int -> !torch.int
          torch.prim.Loop.condition %true, iter(%20 : !torch.int)
        } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
        %13 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
        torch.prim.Loop %1, %true, init()  {
        ^bb0(%arg3: !torch.int):  // no predecessors
          %18 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<!torch.int>, !torch.int -> !torch.int
          %19 = torch.aten.append.t %13, %18 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        %14 = torch.aten.append.t %13, %12 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
        %15 = torch.aten.add.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %16 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
        %17 = torch.aten.__range_length %15, %16, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        torch.prim.Loop %17, %true, init()  {
        ^bb0(%arg3: !torch.int):  // no predecessors
          %18 = torch.aten.__derive_index %arg3, %15, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %19 = torch.aten.__getitem__.t %arg0, %18 : !torch.list<!torch.int>, !torch.int -> !torch.int
          %20 = torch.aten.append.t %13, %19 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %13 : !torch.list<!torch.int>
      }
      torch.prim.If.yield %9 : !torch.list<!torch.int>
    }
    return %7 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.maybe_wrap_dim(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.int {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %0 = torch.aten.le.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.int) {
      torch.prim.If %arg2 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str : !torch.str
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
      torch.prim.RaiseException %str : !torch.str
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
  func @"__torch_mlir_shape_fn.aten.linear"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.linear(%arg0, %arg1, %arg2) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.linear(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>) -> !torch.list<!torch.int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.t(%arg1) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
    %1 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.matmul(%arg0, %0) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    %2 = torch.aten.__isnot__ %arg2, %none : !torch.optional<!torch.list<!torch.int>>, !torch.none -> !torch.bool
    torch.prim.If %2 -> () {
      %3 = torch.prim.unchecked_cast %arg2 : !torch.optional<!torch.list<!torch.int>> -> !torch.list<!torch.int>
      %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%3, %1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
      %5 = torch.aten.eq.int_list %4, %1 : !torch.list<!torch.int>, !torch.list<!torch.int> -> !torch.bool
      torch.prim.If %5 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str : !torch.str
        torch.prim.If.yield
      }
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    return %1 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.matmul(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %int-2 = torch.constant.int -2
    %true = torch.constant.bool true
    %int-1 = torch.constant.int -1
    %str = torch.constant.str "AssertionError: both  arguments to matmul need to be at least 1D"
    %0 = torch.prim.Uninitialized : !torch.list<!torch.int>
    %1 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %2 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %3 = torch.aten.eq.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.bool) {
      %6 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %6 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %5 = torch.prim.If %4 -> (!torch.list<!torch.int>) {
      %6 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.dot(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
      torch.prim.If.yield %6 : !torch.list<!torch.int>
    } else {
      %6 = torch.aten.eq.int %1, %int2 : !torch.int, !torch.int -> !torch.bool
      %7 = torch.prim.If %6 -> (!torch.bool) {
        %9 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %9 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %8 = torch.prim.If %7 -> (!torch.list<!torch.int>) {
        %9 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.mv(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
        torch.prim.If.yield %9 : !torch.list<!torch.int>
      } else {
        %9 = torch.aten.eq.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
        %10 = torch.prim.If %9 -> (!torch.bool) {
          %12 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %12 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %11 = torch.prim.If %10 -> (!torch.list<!torch.int>) {
          %12 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unsqueeze(%arg0, %int0) : (!torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
          %13 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.mm(%12, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
          %14 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.squeeze(%13, %int0) : (!torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
          torch.prim.If.yield %14 : !torch.list<!torch.int>
        } else {
          %12 = torch.aten.eq.int %1, %int2 : !torch.int, !torch.int -> !torch.bool
          %13 = torch.prim.If %12 -> (!torch.bool) {
            %15 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %15 : !torch.bool
          } else {
            torch.prim.If.yield %false : !torch.bool
          }
          %14 = torch.prim.If %13 -> (!torch.list<!torch.int>) {
            %15 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.mm(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
            torch.prim.If.yield %15 : !torch.list<!torch.int>
          } else {
            %15 = torch.aten.ge.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
            %16 = torch.prim.If %15 -> (!torch.bool) {
              %18 = torch.aten.ge.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If.yield %18 : !torch.bool
            } else {
              torch.prim.If.yield %false : !torch.bool
            }
            %17 = torch.prim.If %16 -> (!torch.list<!torch.int>) {
              %18 = torch.aten.gt.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
              %19 = torch.prim.If %18 -> (!torch.int) {
                %28 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<!torch.int>, !torch.int -> !torch.int
                torch.prim.If.yield %28 : !torch.int
              } else {
                torch.prim.If.yield %int1 : !torch.int
              }
              %20 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
              %21 = torch.aten.sub.int %1, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %21, %true, init()  {
              ^bb0(%arg2: !torch.int):  // no predecessors
                %28 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
                %29 = torch.aten.append.t %20, %28 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %22 = torch.aten.__getitem__.t %arg1, %int-1 : !torch.list<!torch.int>, !torch.int -> !torch.int
              %23 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
              %24 = torch.aten.sub.int %2, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %24, %true, init()  {
              ^bb0(%arg2: !torch.int):  // no predecessors
                %28 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
                %29 = torch.aten.append.t %23, %28 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %25 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%20, %23) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
              %26 = torch.aten.gt.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %26 -> () {
                %28 = torch.aten.append.t %25, %19 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              %27 = torch.aten.gt.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %27 -> () {
                %28 = torch.aten.append.t %25, %22 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield %25 : !torch.list<!torch.int>
            } else {
              torch.prim.RaiseException %str : !torch.str
              torch.prim.If.yield %0 : !torch.list<!torch.int>
            }
            torch.prim.If.yield %17 : !torch.list<!torch.int>
          }
          torch.prim.If.yield %14 : !torch.list<!torch.int>
        }
        torch.prim.If.yield %11 : !torch.list<!torch.int>
      }
      torch.prim.If.yield %8 : !torch.list<!torch.int>
    }
    return %5 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.dot(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %7 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
      %8 = torch.aten.eq.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %8 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %4 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %5 = torch.aten.eq.int %3, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %6 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    return %6 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.mv(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      %8 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
      %9 = torch.aten.eq.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %9 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %4 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %5 = torch.aten.eq.int %3, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %6 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %7 = torch.prim.ListConstruct %6 : (!torch.int) -> !torch.list<!torch.int>
    return %7 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.squeeze(%arg0: !torch.list<!torch.int>, %arg1: !torch.int) -> !torch.list<!torch.int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    %1 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.maybe_wrap_dim(%arg1, %1, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    torch.prim.Loop %3, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %4 = torch.aten.eq.int %arg2, %2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %4 -> () {
        %5 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %6 = torch.aten.ne.int %5, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %6 -> () {
          %7 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
          %8 = torch.aten.append.t %0, %7 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %5 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %6 = torch.aten.append.t %0, %5 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.mm(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: self must be a matrix"
    %str_0 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %str_1 = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0 : !torch.str
      torch.prim.If.yield
    }
    %4 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %5 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %6 = torch.aten.eq.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1 : !torch.str
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    return %9 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unsqueeze(%arg0: !torch.list<!torch.int>, %arg1: !torch.int) -> !torch.list<!torch.int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.add.int %0, %int1 : !torch.int, !torch.int -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.maybe_wrap_dim(%arg1, %1, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
    %3 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers._copy(%arg0) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
    torch.operator "aten.insert.t"(%3, %2, %int1) : (!torch.list<!torch.int>, !torch.int, !torch.int) -> ()
    return %3 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_0 = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %2 = torch.prim.max.int %0, %1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    torch.prim.Loop %2, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
      %5 = torch.aten.sub.int %4, %arg2 : !torch.int, !torch.int -> !torch.int
      %6 = torch.aten.sub.int %0, %int1 : !torch.int, !torch.int -> !torch.int
      %7 = torch.aten.sub.int %6, %5 : !torch.int, !torch.int -> !torch.int
      %8 = torch.aten.sub.int %1, %int1 : !torch.int, !torch.int -> !torch.int
      %9 = torch.aten.sub.int %8, %5 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.ge.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
      %11 = torch.prim.If %10 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg0, %7 : !torch.list<!torch.int>, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %12 = torch.aten.ge.int %9, %int0 : !torch.int, !torch.int -> !torch.bool
      %13 = torch.prim.If %12 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg1, %9 : !torch.list<!torch.int>, !torch.int -> !torch.int
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
        torch.prim.RaiseException %21 : !torch.str
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
      %19 = torch.aten.append.t %3, %18 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %3 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.t(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %str = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.le.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.list<!torch.int>) {
      %5 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
      torch.prim.If.yield %5 : !torch.list<!torch.int>
    } else {
      %5 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      %6 = torch.prim.If %5 -> (!torch.list<!torch.int>) {
        %7 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %8 = torch.prim.ListConstruct %7 : (!torch.int) -> !torch.list<!torch.int>
        torch.prim.If.yield %8 : !torch.list<!torch.int>
      } else {
        %7 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %8 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
        torch.prim.If.yield %9 : !torch.list<!torch.int>
      }
      torch.prim.If.yield %6 : !torch.list<!torch.int>
    }
    return %4 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.zeros"(%arg0: !torch.list<!torch.int>, %arg1: !torch.optional<!torch.int>, %arg2: !torch.optional<!torch.int>, %arg3: !torch.optional<!torch.Device>, %arg4: !torch.optional<!torch.bool>) -> !torch.list<!torch.int> {
    return %arg0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.add.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.float) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.sub.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.float) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.mul.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.div.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.__and__.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.minimum"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.maximum"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.bitwise_and.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.threshold_backward"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.float) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.unsqueeze"(%arg0: !torch.list<!torch.int>, %arg1: !torch.int) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.unsqueeze(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.topk"(%arg0: !torch.list<!torch.int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>> {
    %str = torch.constant.str "k ({}) is too big for dimension {} of size {}"
    %str_0 = torch.constant.str "AssertionError: "
    %0 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %1 = torch.aten.le.int %arg1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      %4 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %5 = torch.aten.format(%str, %arg1, %arg2, %4) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
      %6 = torch.aten.add.str %str_0, %5 : !torch.str, !torch.str -> !torch.str
      torch.prim.RaiseException %6 : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten._set_item.t %arg0, %arg2, %arg1 : !torch.list<!torch.int>, !torch.int, !torch.int -> !torch.list<!torch.int>
    %3 = torch.prim.TupleConstruct %arg0, %arg0 : !torch.list<!torch.int>, !torch.list<!torch.int> -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
    return %3 : !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
  }
  func @"__torch_mlir_shape_fn.aten.conv2d"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.list<!torch.int>, %arg6: !torch.int) -> !torch.list<!torch.int> {
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.conv2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.conv2d(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.list<!torch.int>, %arg6: !torch.int) -> !torch.list<!torch.int> {
    %int4 = torch.constant.int 4
    %str = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.conv_output_size(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
    return %4 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.conv_output_size(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.list<!torch.int>, %arg6: !torch.int) -> !torch.list<!torch.int> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.check_shape_forward(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.none
    %1 = torch.aten.len.t %arg5 : !torch.list<!torch.int> -> !torch.int
    %2 = torch.aten.gt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %4 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    %5 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %6 = torch.aten.append.t %4, %5 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
    %7 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %8 = torch.aten.append.t %4, %7 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
    %9 = torch.aten.__range_length %int2, %3, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %9, %true, init()  {
    ^bb0(%arg7: !torch.int):  // no predecessors
      %10 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %11 = torch.prim.If %2 -> (!torch.int) {
        %27 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
        %28 = torch.aten.__getitem__.t %arg5, %27 : !torch.list<!torch.int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %12 = torch.aten.__getitem__.t %arg1, %10 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %13 = torch.aten.sub.int %12, %int1 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.mul.int %11, %13 : !torch.int, !torch.int -> !torch.int
      %15 = torch.aten.add.int %14, %int1 : !torch.int, !torch.int -> !torch.int
      %16 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %17 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
      %18 = torch.aten.__getitem__.t %arg4, %17 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %19 = torch.aten.mul.int %int2, %18 : !torch.int, !torch.int -> !torch.int
      %20 = torch.aten.add.int %16, %19 : !torch.int, !torch.int -> !torch.int
      %21 = torch.aten.sub.int %20, %15 : !torch.int, !torch.int -> !torch.int
      %22 = torch.aten.sub.int %10, %int2 : !torch.int, !torch.int -> !torch.int
      %23 = torch.aten.__getitem__.t %arg3, %22 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %24 = torch.aten.floordiv.int %21, %23 : !torch.int, !torch.int -> !torch.int
      %25 = torch.aten.add.int %24, %int1 : !torch.int, !torch.int -> !torch.int
      %26 = torch.aten.append.t %4, %25 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %4 : !torch.list<!torch.int>
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.check_shape_forward(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.optional<!torch.list<!torch.int>>, %arg3: !torch.list<!torch.int>, %arg4: !torch.list<!torch.int>, %arg5: !torch.list<!torch.int>, %arg6: !torch.int) -> !torch.none {
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %2 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.check_non_negative(%arg4) : (!torch.list<!torch.int>) -> !torch.bool
    %3 = torch.aten.__not__ %2 : !torch.bool -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %4 = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.check_non_negative(%arg3) : (!torch.list<!torch.int>) -> !torch.bool
    %5 = torch.aten.__not__ %4 : !torch.bool -> !torch.bool
    torch.prim.If %5 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %6 = torch.aten.eq.int %1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %8 = torch.aten.ge.int %7, %arg6 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %9 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %10 = torch.aten.remainder.int %9, %arg6 : !torch.int, !torch.int -> !torch.int
    %11 = torch.aten.eq.int %10, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %12 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %13 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %14 = torch.aten.mul.int %13, %arg6 : !torch.int, !torch.int -> !torch.int
    %15 = torch.aten.eq.int %12, %14 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %15 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %16 = torch.aten.__is__ %arg2, %none : !torch.optional<!torch.list<!torch.int>>, !torch.none -> !torch.bool
    %17 = torch.prim.If %16 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %19 = torch.prim.unchecked_cast %arg2 : !torch.optional<!torch.list<!torch.int>> -> !torch.list<!torch.int>
      %20 = torch.aten.len.t %19 : !torch.list<!torch.int> -> !torch.int
      %21 = torch.aten.eq.int %20, %int1 : !torch.int, !torch.int -> !torch.bool
      %22 = torch.prim.If %21 -> (!torch.bool) {
        %23 = torch.aten.__getitem__.t %19, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
        %24 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<!torch.int>, !torch.int -> !torch.int
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
      torch.prim.RaiseException %str : !torch.str
      torch.prim.If.yield
    }
    %18 = torch.aten.__range_length %int2, %0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %18, %true, init()  {
    ^bb0(%arg7: !torch.int):  // no predecessors
      %19 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %20 = torch.aten.__getitem__.t %arg0, %19 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %21 = torch.aten.sub.int %19, %int2 : !torch.int, !torch.int -> !torch.int
      %22 = torch.aten.__getitem__.t %arg4, %21 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %23 = torch.aten.mul.int %int2, %22 : !torch.int, !torch.int -> !torch.int
      %24 = torch.aten.add.int %20, %23 : !torch.int, !torch.int -> !torch.int
      %25 = torch.aten.sub.int %19, %int2 : !torch.int, !torch.int -> !torch.int
      %26 = torch.aten.__getitem__.t %arg5, %25 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %27 = torch.aten.__getitem__.t %arg1, %19 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %28 = torch.aten.sub.int %27, %int1 : !torch.int, !torch.int -> !torch.int
      %29 = torch.aten.mul.int %26, %28 : !torch.int, !torch.int -> !torch.int
      %30 = torch.aten.add.int %29, %int1 : !torch.int, !torch.int -> !torch.int
      %31 = torch.aten.ge.int %24, %30 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %31 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str : !torch.str
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %none : !torch.none
  }
  func @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers.check_non_negative(%arg0: !torch.list<!torch.int>) -> !torch.bool {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%false)  {
    ^bb0(%arg1: !torch.int, %arg2: !torch.bool):  // no predecessors
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<!torch.int>, !torch.int -> !torch.int
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
  func @"__torch_mlir_shape_fn.aten.batch_norm"(%arg0: !torch.list<!torch.int>, %arg1: !torch.optional<!torch.list<!torch.int>>, %arg2: !torch.optional<!torch.list<!torch.int>>, %arg3: !torch.optional<!torch.list<!torch.int>>, %arg4: !torch.optional<!torch.list<!torch.int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool) -> !torch.list<!torch.int> {
    return %arg0 : !torch.list<!torch.int>
  }
}
)mlir");
  return shapeLib;
}
