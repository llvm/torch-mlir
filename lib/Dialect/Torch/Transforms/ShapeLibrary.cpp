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
  func.func @__torch__.torch.jit._shape_functions.unary(%arg0: !torch.list<int>) -> !torch.list<int> {
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
  func.func @__torch__.torch.jit._shape_functions._copy(%arg0: !torch.list<int>) -> !torch.list<int> {
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
  func.func @__torch__.torch.jit._shape_functions.adaptive_avg_pool2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
  func.func @__torch__.torch.jit._shape_functions.zero_dim_tensor(%arg0: !torch.any) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.arange_end(%arg0: !torch.union<float, int>, %arg1: !torch.any, %arg2: !torch.any, %arg3: !torch.any, %arg4: !torch.any) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %0 = torch.operator "aten.ge"(%arg0, %int0) : (!torch.union<float, int>, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.operator "aten.ceil.Scalar"(%arg0) : (!torch.union<float, int>) -> !torch.number
    %2 = torch.aten.Int.Scalar %1 : !torch.number -> !torch.int
    %3 = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
    return %3 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.arange_start(%arg0: !torch.union<float, int>, %arg1: !torch.union<float, int>, %arg2: !torch.any, %arg3: !torch.any, %arg4: !torch.any, %arg5: !torch.any) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %0 = torch.operator "aten.ge"(%arg1, %int0) : (!torch.union<float, int>, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.operator "aten.ge"(%arg1, %arg0) : (!torch.union<float, int>, !torch.union<float, int>) -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.operator "aten.sub"(%arg1, %arg0) : (!torch.union<float, int>, !torch.union<float, int>) -> !torch.number
    %3 = torch.operator "aten.ceil.Scalar"(%2) : (!torch.number) -> !torch.number
    %4 = torch.aten.Int.Scalar %3 : !torch.number -> !torch.int
    %5 = torch.prim.ListConstruct %4 : (!torch.int) -> !torch.list<int>
    return %5 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.arange_start_step(%arg0: !torch.union<float, int>, %arg1: !torch.union<float, int>, %arg2: !torch.union<float, int>, %arg3: !torch.any, %arg4: !torch.any, %arg5: !torch.any, %arg6: !torch.any) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %0 = torch.operator "aten.ne"(%arg2, %int0) : (!torch.union<float, int>, !torch.int) -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = torch.operator "aten.lt"(%arg2, %int0) : (!torch.union<float, int>, !torch.int) -> !torch.bool
    torch.prim.If %1 -> () {
      %6 = torch.operator "aten.ge"(%arg0, %arg1) : (!torch.union<float, int>, !torch.union<float, int>) -> !torch.bool
      torch.prim.If %6 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    } else {
      %6 = torch.operator "aten.ge"(%arg1, %arg0) : (!torch.union<float, int>, !torch.union<float, int>) -> !torch.bool
      torch.prim.If %6 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    }
    %2 = torch.operator "aten.sub"(%arg1, %arg0) : (!torch.union<float, int>, !torch.union<float, int>) -> !torch.number
    %3 = torch.aten.div %2, %arg2 : !torch.number, !torch.union<float, int> -> !torch.float
    %4 = torch.aten.ceil.float %3 : !torch.float -> !torch.int
    %5 = torch.prim.ListConstruct %4 : (!torch.int) -> !torch.list<int>
    return %5 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.squeeze_nodim(%arg0: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
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
  func.func @__torch__.torch.jit._shape_functions.squeeze(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %2 = torch.aten.le.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %1 : !torch.int
    }
    %4 = torch.aten.neg.int %3 : !torch.int -> !torch.int
    %5 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %6 = torch.aten.lt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %12 = torch.aten.gt.int %arg1, %5 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %12 : !torch.bool
    }
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.int) {
      %12 = torch.aten.add.int %arg1, %3 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %12 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %11 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %11, %true, init() {
    ^bb0(%arg2: !torch.int):
      %12 = torch.aten.eq.int %arg2, %10 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %12 -> () {
        %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %14 = torch.aten.ne.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %14 -> () {
          %15 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %16 = torch.aten.append.t %0, %15 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %14 = torch.aten.append.t %0, %13 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.maybe_wrap_dim(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.int {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
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
  func.func @__torch__.torch.jit._shape_functions.unsqueeze(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.add.int %0, %int1 : !torch.int, !torch.int -> !torch.int
    %2 = torch.aten.le.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %1 : !torch.int
    }
    %4 = torch.aten.neg.int %3 : !torch.int -> !torch.int
    %5 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %6 = torch.aten.lt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %13 = torch.aten.gt.int %arg1, %5 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %13 : !torch.bool
    }
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.int) {
      %13 = torch.aten.add.int %arg1, %3 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %13 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %11 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %12 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %12, %true, init() {
    ^bb0(%arg2: !torch.int):
      %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %14 = torch.aten.append.t %11, %13 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    torch.aten.insert.t %11, %10, %int1 : !torch.list<int>, !torch.int, !torch.int
    return %11 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.slice(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.int) -> !torch.list<int> {
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.ne.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %4 = torch.aten.neg.int %3 : !torch.int -> !torch.int
    %5 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %6 = torch.aten.lt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %33 = torch.aten.gt.int %arg1, %5 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %33 : !torch.bool
    }
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.int) {
      %33 = torch.aten.add.int %arg1, %3 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %33 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %11 = torch.aten.__isnot__ %arg2, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %12 = torch.prim.If %11 -> (!torch.int) {
      %33 = torch.prim.unchecked_cast %arg2 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %33 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %13 = torch.aten.__isnot__ %arg3, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %14 = torch.prim.If %13 -> (!torch.int) {
      %33 = torch.prim.unchecked_cast %arg3 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %33 : !torch.int
    } else {
      torch.prim.If.yield %int9223372036854775807 : !torch.int
    }
    %15 = torch.aten.gt.int %arg4, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %15 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %16 = torch.aten.eq.int %12, %int9223372036854775807 : !torch.int, !torch.int -> !torch.bool
    %17 = torch.prim.If %16 -> (!torch.int) {
      torch.prim.If.yield %int0 : !torch.int
    } else {
      torch.prim.If.yield %12 : !torch.int
    }
    %18 = torch.aten.lt.int %17, %int0 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.int) {
      %33 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %34 = torch.aten.add.int %17, %33 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %34 : !torch.int
    } else {
      torch.prim.If.yield %17 : !torch.int
    }
    %20 = torch.aten.lt.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %21 = torch.prim.If %20 -> (!torch.int) {
      %33 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %34 = torch.aten.add.int %14, %33 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %34 : !torch.int
    } else {
      torch.prim.If.yield %14 : !torch.int
    }
    %22 = torch.aten.lt.int %19, %int0 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %int0 : !torch.int
    } else {
      %33 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %34 = torch.aten.gt.int %19, %33 : !torch.int, !torch.int -> !torch.bool
      %35 = torch.prim.If %34 -> (!torch.int) {
        %36 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %36 : !torch.int
      } else {
        torch.prim.If.yield %19 : !torch.int
      }
      torch.prim.If.yield %35 : !torch.int
    }
    %24 = torch.aten.lt.int %21, %23 : !torch.int, !torch.int -> !torch.bool
    %25 = torch.prim.If %24 -> (!torch.int) {
      torch.prim.If.yield %23 : !torch.int
    } else {
      %33 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
      %34 = torch.aten.ge.int %21, %33 : !torch.int, !torch.int -> !torch.bool
      %35 = torch.prim.If %34 -> (!torch.int) {
        %36 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %36 : !torch.int
      } else {
        torch.prim.If.yield %21 : !torch.int
      }
      torch.prim.If.yield %35 : !torch.int
    }
    %26 = torch.aten.sub.int %25, %23 : !torch.int, !torch.int -> !torch.int
    %27 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %28 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %28, %true, init() {
    ^bb0(%arg5: !torch.int):
      %33 = torch.aten.__getitem__.t %arg0, %arg5 : !torch.list<int>, !torch.int -> !torch.int
      %34 = torch.aten.append.t %27, %33 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %29 = torch.aten.add.int %26, %arg4 : !torch.int, !torch.int -> !torch.int
    %30 = torch.aten.sub.int %29, %int1 : !torch.int, !torch.int -> !torch.int
    %31 = torch.aten.floordiv.int %30, %arg4 : !torch.int, !torch.int -> !torch.int
    %32 = torch.aten._set_item.t %27, %10, %31 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    return %27 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.max_int() -> !torch.int {
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    return %int9223372036854775807 : !torch.int
  }
  func.func @__torch__.torch.jit._shape_functions.select(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.ne.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %4 = torch.aten.neg.int %3 : !torch.int -> !torch.int
    %5 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %6 = torch.aten.lt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %17 = torch.aten.gt.int %arg1, %5 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %17 : !torch.bool
    }
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.int) {
      %17 = torch.aten.add.int %arg1, %3 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %17 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %11 = torch.aten.__getitem__.t %arg0, %10 : !torch.list<int>, !torch.int -> !torch.int
    %12 = torch.aten.neg.int %11 : !torch.int -> !torch.int
    %13 = torch.aten.lt.int %arg2, %12 : !torch.int, !torch.int -> !torch.bool
    %14 = torch.prim.If %13 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %17 = torch.aten.ge.int %arg2, %11 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %17 : !torch.bool
    }
    %15 = torch.aten.__not__ %14 : !torch.bool -> !torch.bool
    torch.prim.If %15 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %0, %true, init() {
    ^bb0(%arg3: !torch.int):
      %17 = torch.aten.ne.int %arg3, %10 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %17 -> () {
        %18 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %19 = torch.aten.append.t %16, %18 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %16 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.index_select(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %3 = torch.aten.neg.int %2 : !torch.int -> !torch.int
    %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.lt.int %arg1, %3 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %18 = torch.aten.gt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %18 : !torch.bool
    }
    %7 = torch.aten.__not__ %6 : !torch.bool -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.int) {
      %18 = torch.aten.add.int %arg1, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %18 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %10 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %11 = torch.prim.Loop %10, %true, init(%int1) {
    ^bb0(%arg3: !torch.int, %arg4: !torch.int):
      %18 = torch.aten.__getitem__.t %arg2, %arg3 : !torch.list<int>, !torch.int -> !torch.int
      %19 = torch.aten.mul.int %arg4, %18 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%19 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    %12 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %13 = torch.aten.le.int %12, %int1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %13 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %14 = torch.aten.eq.int %9, %int0 : !torch.int, !torch.int -> !torch.bool
    %15 = torch.prim.If %14 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %18 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %19 = torch.aten.lt.int %9, %18 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %19 : !torch.bool
    }
    torch.prim.If %15 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %17 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %17, %true, init() {
    ^bb0(%arg3: !torch.int):
      %18 = torch.aten.eq.int %9, %arg3 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %18 -> () {
        %19 = torch.aten.append.t %16, %11 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      } else {
        %19 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.append.t %16, %19 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %16 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.multiply_integers(%arg0: !torch.list<int>) -> !torch.int {
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.mul.int %arg2, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%3 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    return %1 : !torch.int
  }
  func.func @__torch__.torch.jit._shape_functions.embedding(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.list<int> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
      %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %6 = torch.aten.le.int %5, %int0 : !torch.int, !torch.int -> !torch.bool
      %7 = torch.prim.If %6 -> (!torch.int) {
        torch.prim.If.yield %int1 : !torch.int
      } else {
        torch.prim.If.yield %5 : !torch.int
      }
      %8 = torch.aten.neg.int %7 : !torch.int -> !torch.int
      %9 = torch.aten.sub.int %7, %int1 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.lt.int %int0, %8 : !torch.int, !torch.int -> !torch.bool
      %11 = torch.prim.If %10 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        %19 = torch.aten.gt.int %int0, %9 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %19 : !torch.bool
      }
      %12 = torch.aten.__not__ %11 : !torch.bool -> !torch.bool
      torch.prim.If %12 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %13 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %14 = torch.prim.Loop %13, %true, init(%int1) {
      ^bb0(%arg5: !torch.int, %arg6: !torch.int):
        %19 = torch.aten.__getitem__.t %arg1, %arg5 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.mul.int %arg6, %19 : !torch.int, !torch.int -> !torch.int
        torch.prim.Loop.condition %true, iter(%20 : !torch.int)
      } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
      %15 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %16 = torch.aten.le.int %15, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %16 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %17 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %18 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      torch.prim.Loop %18, %true, init() {
      ^bb0(%arg5: !torch.int):
        %19 = torch.aten.eq.int %int0, %arg5 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %19 -> () {
          %20 = torch.aten.append.t %17, %14 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          %20 = torch.aten.__getitem__.t %arg0, %arg5 : !torch.list<int>, !torch.int -> !torch.int
          %21 = torch.aten.append.t %17, %20 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        }
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %17 : !torch.list<int>
    } else {
      %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %6 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      torch.prim.Loop %6, %true, init() {
      ^bb0(%arg5: !torch.int):
        %9 = torch.aten.__getitem__.t %arg1, %arg5 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.append.t %5, %9 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      %7 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %8 = torch.aten.append.t %5, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    }
    return %4 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.mm(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: "
    %str_0 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: self must be a matrix"
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
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
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<int>
    return %9 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.dot(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
  func.func @__torch__.torch.jit._shape_functions.mv(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %false = torch.constant.bool false
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
  func.func @__torch__.torch.jit._shape_functions.matmul(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_0 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %str_1 = torch.constant.str "AssertionError: self must be a matrix"
    %str_2 = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %str_3 = torch.constant.str "AssertionError: both  arguments to matmul need to be at least 1D"
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %int-2 = torch.constant.int -2
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
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
      %6 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %7 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %8 = torch.aten.eq.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.bool) {
        %13 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
        %14 = torch.aten.eq.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %14 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %9 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %10 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %11 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %12 = torch.aten.eq.int %10, %11 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %12 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
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
        %9 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %10 = torch.aten.eq.int %9, %int2 : !torch.int, !torch.int -> !torch.bool
        %11 = torch.prim.If %10 -> (!torch.bool) {
          %17 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
          %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %18 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        torch.prim.If %11 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %12 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
        %13 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %14 = torch.aten.eq.int %12, %13 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %14 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %15 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %16 = torch.prim.ListConstruct %15 : (!torch.int) -> !torch.list<int>
        torch.prim.If.yield %16 : !torch.list<int>
      } else {
        %9 = torch.aten.eq.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
        %10 = torch.prim.If %9 -> (!torch.bool) {
          %12 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %12 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %11 = torch.prim.If %10 -> (!torch.list<int>) {
          %12 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
          %13 = torch.aten.add.int %12, %int1 : !torch.int, !torch.int -> !torch.int
          %14 = torch.aten.le.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
          %15 = torch.prim.If %14 -> (!torch.int) {
            torch.prim.If.yield %int1 : !torch.int
          } else {
            torch.prim.If.yield %13 : !torch.int
          }
          %16 = torch.aten.neg.int %15 : !torch.int -> !torch.int
          %17 = torch.aten.sub.int %15, %int1 : !torch.int, !torch.int -> !torch.int
          %18 = torch.aten.lt.int %int0, %16 : !torch.int, !torch.int -> !torch.bool
          %19 = torch.prim.If %18 -> (!torch.bool) {
            torch.prim.If.yield %true : !torch.bool
          } else {
            %34 = torch.aten.gt.int %int0, %17 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %34 : !torch.bool
          }
          %20 = torch.aten.__not__ %19 : !torch.bool -> !torch.bool
          torch.prim.If %20 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %21 = torch.prim.ListConstruct  : () -> !torch.list<int>
          %22 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
          torch.prim.Loop %22, %true, init() {
          ^bb0(%arg2: !torch.int):
            %34 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
            %35 = torch.aten.append.t %21, %34 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.Loop.condition %true, iter()
          } : (!torch.int, !torch.bool) -> ()
          torch.aten.insert.t %21, %int0, %int1 : !torch.list<int>, !torch.int, !torch.int
          %23 = torch.aten.len.t %21 : !torch.list<int> -> !torch.int
          %24 = torch.aten.eq.int %23, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %24 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %25 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
          %26 = torch.aten.eq.int %25, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %26 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %27 = torch.aten.__getitem__.t %21, %int1 : !torch.list<int>, !torch.int -> !torch.int
          %28 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
          %29 = torch.aten.eq.int %27, %28 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %29 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %30 = torch.aten.__getitem__.t %21, %int0 : !torch.list<int>, !torch.int -> !torch.int
          %31 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
          %32 = torch.prim.ListConstruct %30, %31 : (!torch.int, !torch.int) -> !torch.list<int>
          %33 = torch.prim.ListConstruct  : () -> !torch.list<int>
          torch.prim.Loop %int2, %true, init() {
          ^bb0(%arg2: !torch.int):
            %34 = torch.aten.eq.int %arg2, %int0 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %34 -> () {
              %35 = torch.aten.__getitem__.t %32, %arg2 : !torch.list<int>, !torch.int -> !torch.int
              %36 = torch.aten.ne.int %35, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %36 -> () {
                %37 = torch.aten.__getitem__.t %32, %arg2 : !torch.list<int>, !torch.int -> !torch.int
                %38 = torch.aten.append.t %33, %37 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield
            } else {
              %35 = torch.aten.__getitem__.t %32, %arg2 : !torch.list<int>, !torch.int -> !torch.int
              %36 = torch.aten.append.t %33, %35 : !torch.list<int>, !torch.int -> !torch.list<int>
              torch.prim.If.yield
            }
            torch.prim.Loop.condition %true, iter()
          } : (!torch.int, !torch.bool) -> ()
          torch.prim.If.yield %33 : !torch.list<int>
        } else {
          %12 = torch.aten.eq.int %1, %int2 : !torch.int, !torch.int -> !torch.bool
          %13 = torch.prim.If %12 -> (!torch.bool) {
            %15 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %15 : !torch.bool
          } else {
            torch.prim.If.yield %false : !torch.bool
          }
          %14 = torch.prim.If %13 -> (!torch.list<int>) {
            %15 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
            %16 = torch.aten.eq.int %15, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %16 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %17 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
            %18 = torch.aten.eq.int %17, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %18 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %19 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
            %20 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
            %21 = torch.aten.eq.int %19, %20 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %21 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %22 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
            %23 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
            %24 = torch.prim.ListConstruct %22, %23 : (!torch.int, !torch.int) -> !torch.list<int>
            torch.prim.If.yield %24 : !torch.list<int>
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
                %31 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
                torch.prim.If.yield %31 : !torch.int
              } else {
                torch.prim.If.yield %int1 : !torch.int
              }
              %20 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %21 = torch.aten.sub.int %1, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %21, %true, init() {
              ^bb0(%arg2: !torch.int):
                %31 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
                %32 = torch.aten.append.t %20, %31 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %22 = torch.aten.__getitem__.t %arg1, %int-1 : !torch.list<int>, !torch.int -> !torch.int
              %23 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %24 = torch.aten.sub.int %2, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %24, %true, init() {
              ^bb0(%arg2: !torch.int):
                %31 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
                %32 = torch.aten.append.t %23, %31 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %25 = torch.aten.len.t %20 : !torch.list<int> -> !torch.int
              %26 = torch.aten.len.t %23 : !torch.list<int> -> !torch.int
              %27 = torch.prim.max.int %25, %26 : !torch.int, !torch.int -> !torch.int
              %28 = torch.prim.ListConstruct  : () -> !torch.list<int>
              torch.prim.Loop %27, %true, init() {
              ^bb0(%arg2: !torch.int):
                %31 = torch.aten.sub.int %27, %int1 : !torch.int, !torch.int -> !torch.int
                %32 = torch.aten.sub.int %31, %arg2 : !torch.int, !torch.int -> !torch.int
                %33 = torch.aten.sub.int %25, %int1 : !torch.int, !torch.int -> !torch.int
                %34 = torch.aten.sub.int %33, %32 : !torch.int, !torch.int -> !torch.int
                %35 = torch.aten.sub.int %26, %int1 : !torch.int, !torch.int -> !torch.int
                %36 = torch.aten.sub.int %35, %32 : !torch.int, !torch.int -> !torch.int
                %37 = torch.aten.ge.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
                %38 = torch.prim.If %37 -> (!torch.int) {
                  %47 = torch.aten.__getitem__.t %20, %34 : !torch.list<int>, !torch.int -> !torch.int
                  torch.prim.If.yield %47 : !torch.int
                } else {
                  torch.prim.If.yield %int1 : !torch.int
                }
                %39 = torch.aten.ge.int %36, %int0 : !torch.int, !torch.int -> !torch.bool
                %40 = torch.prim.If %39 -> (!torch.int) {
                  %47 = torch.aten.__getitem__.t %23, %36 : !torch.list<int>, !torch.int -> !torch.int
                  torch.prim.If.yield %47 : !torch.int
                } else {
                  torch.prim.If.yield %int1 : !torch.int
                }
                %41 = torch.aten.ne.int %38, %40 : !torch.int, !torch.int -> !torch.bool
                %42 = torch.prim.If %41 -> (!torch.bool) {
                  %47 = torch.aten.ne.int %38, %int1 : !torch.int, !torch.int -> !torch.bool
                  torch.prim.If.yield %47 : !torch.bool
                } else {
                  torch.prim.If.yield %false : !torch.bool
                }
                %43 = torch.prim.If %42 -> (!torch.bool) {
                  %47 = torch.aten.ne.int %40, %int1 : !torch.int, !torch.int -> !torch.bool
                  torch.prim.If.yield %47 : !torch.bool
                } else {
                  torch.prim.If.yield %false : !torch.bool
                }
                torch.prim.If %43 -> () {
                  %47 = torch.aten.format(%str, %38, %40, %arg2) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
                  %48 = torch.aten.add.str %str_2, %47 : !torch.str, !torch.str -> !torch.str
                  torch.prim.RaiseException %48, %none : !torch.str, !torch.none
                  torch.prim.If.yield
                } else {
                  torch.prim.If.yield
                }
                %44 = torch.aten.eq.int %38, %int1 : !torch.int, !torch.int -> !torch.bool
                %45 = torch.prim.If %44 -> (!torch.int) {
                  torch.prim.If.yield %40 : !torch.int
                } else {
                  torch.prim.If.yield %38 : !torch.int
                }
                %46 = torch.aten.append.t %28, %45 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %29 = torch.aten.gt.int %1, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %29 -> () {
                %31 = torch.aten.append.t %28, %19 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              %30 = torch.aten.gt.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %30 -> () {
                %31 = torch.aten.append.t %28, %22 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield %28 : !torch.list<int>
            } else {
              torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
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
  func.func @__torch__.torch.jit._shape_functions.broadcast(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %str_0 = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
        %20 = torch.aten.format(%str_0, %11, %13, %arg2) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %21 = torch.aten.add.str %str, %20 : !torch.str, !torch.str -> !torch.str
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
  func.func @__torch__.torch.jit._shape_functions.linear(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: both  arguments to matmul need to be at least 1D"
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %int-2 = torch.constant.int -2
    %false = torch.constant.bool false
    %str_0 = torch.constant.str "AssertionError: self must be a matrix"
    %str_1 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %str_2 = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_3 = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.le.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.prim.If %3 -> (!torch.list<int>) {
      %13 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %13 : !torch.list<int>
    } else {
      %13 = torch.aten.eq.int %2, %int1 : !torch.int, !torch.int -> !torch.bool
      %14 = torch.prim.If %13 -> (!torch.list<int>) {
        %15 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %16 = torch.prim.ListConstruct %15 : (!torch.int) -> !torch.list<int>
        torch.prim.If.yield %16 : !torch.list<int>
      } else {
        %15 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
        %16 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %17 = torch.prim.ListConstruct %15, %16 : (!torch.int, !torch.int) -> !torch.list<int>
        torch.prim.If.yield %17 : !torch.list<int>
      }
      torch.prim.If.yield %14 : !torch.list<int>
    }
    %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %6 = torch.prim.Uninitialized : !torch.list<int>
    %7 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %8 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
    %9 = torch.aten.eq.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
    %10 = torch.prim.If %9 -> (!torch.bool) {
      %13 = torch.aten.eq.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %13 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %11 = torch.prim.If %10 -> (!torch.list<int>) {
      %13 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %14 = torch.aten.eq.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.bool) {
        %19 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
        %20 = torch.aten.eq.int %19, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %15 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %16 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %17 = torch.aten.__getitem__.t %4, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %18 = torch.aten.eq.int %16, %17 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %18 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %13 = torch.aten.eq.int %7, %int2 : !torch.int, !torch.int -> !torch.bool
      %14 = torch.prim.If %13 -> (!torch.bool) {
        %16 = torch.aten.eq.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %16 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %15 = torch.prim.If %14 -> (!torch.list<int>) {
        %16 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %17 = torch.aten.eq.int %16, %int2 : !torch.int, !torch.int -> !torch.bool
        %18 = torch.prim.If %17 -> (!torch.bool) {
          %24 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
          %25 = torch.aten.eq.int %24, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %25 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        torch.prim.If %18 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %19 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.__getitem__.t %4, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %21 = torch.aten.eq.int %19, %20 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %21 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %22 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %23 = torch.prim.ListConstruct %22 : (!torch.int) -> !torch.list<int>
        torch.prim.If.yield %23 : !torch.list<int>
      } else {
        %16 = torch.aten.eq.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
        %17 = torch.prim.If %16 -> (!torch.bool) {
          %19 = torch.aten.eq.int %8, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %19 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %18 = torch.prim.If %17 -> (!torch.list<int>) {
          %19 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
          %20 = torch.aten.add.int %19, %int1 : !torch.int, !torch.int -> !torch.int
          %21 = torch.aten.le.int %20, %int0 : !torch.int, !torch.int -> !torch.bool
          %22 = torch.prim.If %21 -> (!torch.int) {
            torch.prim.If.yield %int1 : !torch.int
          } else {
            torch.prim.If.yield %20 : !torch.int
          }
          %23 = torch.aten.neg.int %22 : !torch.int -> !torch.int
          %24 = torch.aten.sub.int %22, %int1 : !torch.int, !torch.int -> !torch.int
          %25 = torch.aten.lt.int %int0, %23 : !torch.int, !torch.int -> !torch.bool
          %26 = torch.prim.If %25 -> (!torch.bool) {
            torch.prim.If.yield %true : !torch.bool
          } else {
            %41 = torch.aten.gt.int %int0, %24 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %41 : !torch.bool
          }
          %27 = torch.aten.__not__ %26 : !torch.bool -> !torch.bool
          torch.prim.If %27 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %28 = torch.prim.ListConstruct  : () -> !torch.list<int>
          %29 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
          torch.prim.Loop %29, %true, init() {
          ^bb0(%arg3: !torch.int):
            %41 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
            %42 = torch.aten.append.t %28, %41 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.Loop.condition %true, iter()
          } : (!torch.int, !torch.bool) -> ()
          torch.aten.insert.t %28, %int0, %int1 : !torch.list<int>, !torch.int, !torch.int
          %30 = torch.aten.len.t %28 : !torch.list<int> -> !torch.int
          %31 = torch.aten.eq.int %30, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %31 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %32 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
          %33 = torch.aten.eq.int %32, %int2 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %33 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %34 = torch.aten.__getitem__.t %28, %int1 : !torch.list<int>, !torch.int -> !torch.int
          %35 = torch.aten.__getitem__.t %4, %int0 : !torch.list<int>, !torch.int -> !torch.int
          %36 = torch.aten.eq.int %34, %35 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %36 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %37 = torch.aten.__getitem__.t %28, %int0 : !torch.list<int>, !torch.int -> !torch.int
          %38 = torch.aten.__getitem__.t %4, %int1 : !torch.list<int>, !torch.int -> !torch.int
          %39 = torch.prim.ListConstruct %37, %38 : (!torch.int, !torch.int) -> !torch.list<int>
          %40 = torch.prim.ListConstruct  : () -> !torch.list<int>
          torch.prim.Loop %int2, %true, init() {
          ^bb0(%arg3: !torch.int):
            %41 = torch.aten.eq.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %41 -> () {
              %42 = torch.aten.__getitem__.t %39, %arg3 : !torch.list<int>, !torch.int -> !torch.int
              %43 = torch.aten.ne.int %42, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %43 -> () {
                %44 = torch.aten.__getitem__.t %39, %arg3 : !torch.list<int>, !torch.int -> !torch.int
                %45 = torch.aten.append.t %40, %44 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield
            } else {
              %42 = torch.aten.__getitem__.t %39, %arg3 : !torch.list<int>, !torch.int -> !torch.int
              %43 = torch.aten.append.t %40, %42 : !torch.list<int>, !torch.int -> !torch.list<int>
              torch.prim.If.yield
            }
            torch.prim.Loop.condition %true, iter()
          } : (!torch.int, !torch.bool) -> ()
          torch.prim.If.yield %40 : !torch.list<int>
        } else {
          %19 = torch.aten.eq.int %7, %int2 : !torch.int, !torch.int -> !torch.bool
          %20 = torch.prim.If %19 -> (!torch.bool) {
            %22 = torch.aten.eq.int %8, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %22 : !torch.bool
          } else {
            torch.prim.If.yield %false : !torch.bool
          }
          %21 = torch.prim.If %20 -> (!torch.list<int>) {
            %22 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
            %23 = torch.aten.eq.int %22, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %23 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %24 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
            %25 = torch.aten.eq.int %24, %int2 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %25 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %26 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
            %27 = torch.aten.__getitem__.t %4, %int0 : !torch.list<int>, !torch.int -> !torch.int
            %28 = torch.aten.eq.int %26, %27 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %28 -> () {
              torch.prim.If.yield
            } else {
              torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
              torch.prim.If.yield
            }
            %29 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
            %30 = torch.aten.__getitem__.t %4, %int1 : !torch.list<int>, !torch.int -> !torch.int
            %31 = torch.prim.ListConstruct %29, %30 : (!torch.int, !torch.int) -> !torch.list<int>
            torch.prim.If.yield %31 : !torch.list<int>
          } else {
            %22 = torch.aten.ge.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
            %23 = torch.prim.If %22 -> (!torch.bool) {
              %25 = torch.aten.ge.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If.yield %25 : !torch.bool
            } else {
              torch.prim.If.yield %false : !torch.bool
            }
            %24 = torch.prim.If %23 -> (!torch.list<int>) {
              %25 = torch.aten.gt.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
              %26 = torch.prim.If %25 -> (!torch.int) {
                %38 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
                torch.prim.If.yield %38 : !torch.int
              } else {
                torch.prim.If.yield %int1 : !torch.int
              }
              %27 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %28 = torch.aten.sub.int %7, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %28, %true, init() {
              ^bb0(%arg3: !torch.int):
                %38 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
                %39 = torch.aten.append.t %27, %38 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %29 = torch.aten.__getitem__.t %4, %int-1 : !torch.list<int>, !torch.int -> !torch.int
              %30 = torch.prim.ListConstruct  : () -> !torch.list<int>
              %31 = torch.aten.sub.int %8, %int2 : !torch.int, !torch.int -> !torch.int
              torch.prim.Loop %31, %true, init() {
              ^bb0(%arg3: !torch.int):
                %38 = torch.aten.__getitem__.t %4, %arg3 : !torch.list<int>, !torch.int -> !torch.int
                %39 = torch.aten.append.t %30, %38 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %32 = torch.aten.len.t %27 : !torch.list<int> -> !torch.int
              %33 = torch.aten.len.t %30 : !torch.list<int> -> !torch.int
              %34 = torch.prim.max.int %32, %33 : !torch.int, !torch.int -> !torch.int
              %35 = torch.prim.ListConstruct  : () -> !torch.list<int>
              torch.prim.Loop %34, %true, init() {
              ^bb0(%arg3: !torch.int):
                %38 = torch.aten.sub.int %34, %int1 : !torch.int, !torch.int -> !torch.int
                %39 = torch.aten.sub.int %38, %arg3 : !torch.int, !torch.int -> !torch.int
                %40 = torch.aten.sub.int %32, %int1 : !torch.int, !torch.int -> !torch.int
                %41 = torch.aten.sub.int %40, %39 : !torch.int, !torch.int -> !torch.int
                %42 = torch.aten.sub.int %33, %int1 : !torch.int, !torch.int -> !torch.int
                %43 = torch.aten.sub.int %42, %39 : !torch.int, !torch.int -> !torch.int
                %44 = torch.aten.ge.int %41, %int0 : !torch.int, !torch.int -> !torch.bool
                %45 = torch.prim.If %44 -> (!torch.int) {
                  %54 = torch.aten.__getitem__.t %27, %41 : !torch.list<int>, !torch.int -> !torch.int
                  torch.prim.If.yield %54 : !torch.int
                } else {
                  torch.prim.If.yield %int1 : !torch.int
                }
                %46 = torch.aten.ge.int %43, %int0 : !torch.int, !torch.int -> !torch.bool
                %47 = torch.prim.If %46 -> (!torch.int) {
                  %54 = torch.aten.__getitem__.t %30, %43 : !torch.list<int>, !torch.int -> !torch.int
                  torch.prim.If.yield %54 : !torch.int
                } else {
                  torch.prim.If.yield %int1 : !torch.int
                }
                %48 = torch.aten.ne.int %45, %47 : !torch.int, !torch.int -> !torch.bool
                %49 = torch.prim.If %48 -> (!torch.bool) {
                  %54 = torch.aten.ne.int %45, %int1 : !torch.int, !torch.int -> !torch.bool
                  torch.prim.If.yield %54 : !torch.bool
                } else {
                  torch.prim.If.yield %false : !torch.bool
                }
                %50 = torch.prim.If %49 -> (!torch.bool) {
                  %54 = torch.aten.ne.int %47, %int1 : !torch.int, !torch.int -> !torch.bool
                  torch.prim.If.yield %54 : !torch.bool
                } else {
                  torch.prim.If.yield %false : !torch.bool
                }
                torch.prim.If %50 -> () {
                  %54 = torch.aten.format(%str_2, %45, %47, %arg3) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
                  %55 = torch.aten.add.str %str_3, %54 : !torch.str, !torch.str -> !torch.str
                  torch.prim.RaiseException %55, %none : !torch.str, !torch.none
                  torch.prim.If.yield
                } else {
                  torch.prim.If.yield
                }
                %51 = torch.aten.eq.int %45, %int1 : !torch.int, !torch.int -> !torch.bool
                %52 = torch.prim.If %51 -> (!torch.int) {
                  torch.prim.If.yield %47 : !torch.int
                } else {
                  torch.prim.If.yield %45 : !torch.int
                }
                %53 = torch.aten.append.t %35, %52 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.Loop.condition %true, iter()
              } : (!torch.int, !torch.bool) -> ()
              %36 = torch.aten.gt.int %7, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %36 -> () {
                %38 = torch.aten.append.t %35, %26 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              %37 = torch.aten.gt.int %8, %int1 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %37 -> () {
                %38 = torch.aten.append.t %35, %29 : !torch.list<int>, !torch.int -> !torch.list<int>
                torch.prim.If.yield
              } else {
                torch.prim.If.yield
              }
              torch.prim.If.yield %35 : !torch.list<int>
            } else {
              torch.prim.RaiseException %str, %none : !torch.str, !torch.none
              torch.prim.If.yield %6 : !torch.list<int>
            }
            torch.prim.If.yield %24 : !torch.list<int>
          }
          torch.prim.If.yield %21 : !torch.list<int>
        }
        torch.prim.If.yield %18 : !torch.list<int>
      }
      torch.prim.If.yield %15 : !torch.list<int>
    }
    %12 = torch.aten.__isnot__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    torch.prim.If %12 -> () {
      %13 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %14 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
      %15 = torch.aten.len.t %11 : !torch.list<int> -> !torch.int
      %16 = torch.prim.max.int %14, %15 : !torch.int, !torch.int -> !torch.int
      %17 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %16, %true, init() {
      ^bb0(%arg3: !torch.int):
        %19 = torch.aten.sub.int %16, %int1 : !torch.int, !torch.int -> !torch.int
        %20 = torch.aten.sub.int %19, %arg3 : !torch.int, !torch.int -> !torch.int
        %21 = torch.aten.sub.int %14, %int1 : !torch.int, !torch.int -> !torch.int
        %22 = torch.aten.sub.int %21, %20 : !torch.int, !torch.int -> !torch.int
        %23 = torch.aten.sub.int %15, %int1 : !torch.int, !torch.int -> !torch.int
        %24 = torch.aten.sub.int %23, %20 : !torch.int, !torch.int -> !torch.int
        %25 = torch.aten.ge.int %22, %int0 : !torch.int, !torch.int -> !torch.bool
        %26 = torch.prim.If %25 -> (!torch.int) {
          %35 = torch.aten.__getitem__.t %13, %22 : !torch.list<int>, !torch.int -> !torch.int
          torch.prim.If.yield %35 : !torch.int
        } else {
          torch.prim.If.yield %int1 : !torch.int
        }
        %27 = torch.aten.ge.int %24, %int0 : !torch.int, !torch.int -> !torch.bool
        %28 = torch.prim.If %27 -> (!torch.int) {
          %35 = torch.aten.__getitem__.t %11, %24 : !torch.list<int>, !torch.int -> !torch.int
          torch.prim.If.yield %35 : !torch.int
        } else {
          torch.prim.If.yield %int1 : !torch.int
        }
        %29 = torch.aten.ne.int %26, %28 : !torch.int, !torch.int -> !torch.bool
        %30 = torch.prim.If %29 -> (!torch.bool) {
          %35 = torch.aten.ne.int %26, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %35 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %31 = torch.prim.If %30 -> (!torch.bool) {
          %35 = torch.aten.ne.int %28, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %35 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        torch.prim.If %31 -> () {
          %35 = torch.aten.format(%str_2, %26, %28, %arg3) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
          %36 = torch.aten.add.str %str_3, %35 : !torch.str, !torch.str -> !torch.str
          torch.prim.RaiseException %36, %none : !torch.str, !torch.none
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        %32 = torch.aten.eq.int %26, %int1 : !torch.int, !torch.int -> !torch.bool
        %33 = torch.prim.If %32 -> (!torch.int) {
          torch.prim.If.yield %28 : !torch.int
        } else {
          torch.prim.If.yield %26 : !torch.int
        }
        %34 = torch.aten.append.t %17, %33 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      %18 = torch.aten.eq.int_list %17, %11 : !torch.list<int>, !torch.list<int> -> !torch.bool
      torch.prim.If %18 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    return %11 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.t(%arg0: !torch.list<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
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
  func.func @__torch__.torch.jit._shape_functions.max_pool2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.list<int> {
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: stride should not be zeero"
    %int-1 = torch.constant.int -1
    %int-2 = torch.constant.int -2
    %int-3 = torch.constant.int -3
    %int-4 = torch.constant.int -4
    %str_0 = torch.constant.str "AssertionError: "
    %str_1 = torch.constant.str "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
    %str_2 = torch.constant.str "AssertionError: max_pool2d: padding must be either be a single int, or a tuple of two ints"
    %str_3 = torch.constant.str "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    %none = torch.constant.none
    %str_4 = torch.constant.str "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %5 = torch.aten.eq.int %4, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %86 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    }
    %7 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    %10 = torch.prim.If %9 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    torch.prim.If %10 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %11 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %86 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    }
    %14 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %15 = torch.aten.eq.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %6 : !torch.int
    } else {
      %86 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int1 : !torch.int, !torch.int -> !torch.bool
      %88 = torch.prim.If %87 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        %89 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %89 : !torch.int
      }
      torch.prim.If.yield %88 : !torch.int
    }
    %17 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    torch.prim.If %19 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %20 = torch.aten.__getitem__.t %arg3, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %21 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %20 : !torch.int
    } else {
      %86 = torch.aten.__getitem__.t %arg3, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    }
    %24 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %25 = torch.aten.eq.int %24, %int1 : !torch.int, !torch.int -> !torch.bool
    %26 = torch.prim.If %25 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    torch.prim.If %26 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %27 = torch.aten.__getitem__.t %arg4, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %28 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %29 = torch.aten.eq.int %28, %int1 : !torch.int, !torch.int -> !torch.bool
    %30 = torch.prim.If %29 -> (!torch.int) {
      torch.prim.If.yield %27 : !torch.int
    } else {
      %86 = torch.aten.__getitem__.t %arg4, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    }
    %31 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %32 = torch.aten.eq.int %31, %int3 : !torch.int, !torch.int -> !torch.bool
    %33 = torch.prim.If %32 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %87 = torch.aten.eq.int %86, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    }
    torch.prim.If %33 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %34 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %35 = torch.aten.eq.int %34, %int4 : !torch.int, !torch.int -> !torch.bool
    %36 = torch.prim.If %35 -> (!torch.int) {
      %86 = torch.aten.__getitem__.t %arg0, %int-4 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    } else {
      torch.prim.If.yield %int1 : !torch.int
    }
    %37 = torch.aten.__getitem__.t %arg0, %int-3 : !torch.list<int>, !torch.int -> !torch.int
    %38 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
    %39 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
    %40 = torch.aten.ne.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %40 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %41 = torch.aten.add.int %38, %20 : !torch.int, !torch.int -> !torch.int
    %42 = torch.aten.add.int %41, %20 : !torch.int, !torch.int -> !torch.int
    %43 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %44 = torch.aten.mul.int %27, %43 : !torch.int, !torch.int -> !torch.int
    %45 = torch.aten.sub.int %42, %44 : !torch.int, !torch.int -> !torch.int
    %46 = torch.aten.sub.int %45, %int1 : !torch.int, !torch.int -> !torch.int
    %47 = torch.prim.If %arg5 -> (!torch.int) {
      %86 = torch.aten.sub.int %13, %int1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %48 = torch.aten.add.int %46, %47 : !torch.int, !torch.int -> !torch.int
    %49 = torch.aten.floordiv.int %48, %13 : !torch.int, !torch.int -> !torch.int
    %50 = torch.aten.add.int %49, %int1 : !torch.int, !torch.int -> !torch.int
    %51 = torch.prim.If %arg5 -> (!torch.int) {
      %86 = torch.aten.mul.int %49, %13 : !torch.int, !torch.int -> !torch.int
      %87 = torch.aten.add.int %38, %20 : !torch.int, !torch.int -> !torch.int
      %88 = torch.aten.ge.int %86, %87 : !torch.int, !torch.int -> !torch.bool
      %89 = torch.prim.If %88 -> (!torch.int) {
        torch.prim.If.yield %49 : !torch.int
      } else {
        torch.prim.If.yield %50 : !torch.int
      }
      torch.prim.If.yield %89 : !torch.int
    } else {
      torch.prim.If.yield %50 : !torch.int
    }
    %52 = torch.aten.ne.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %52 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %53 = torch.aten.add.int %39, %23 : !torch.int, !torch.int -> !torch.int
    %54 = torch.aten.add.int %53, %23 : !torch.int, !torch.int -> !torch.int
    %55 = torch.aten.sub.int %6, %int1 : !torch.int, !torch.int -> !torch.int
    %56 = torch.aten.mul.int %30, %55 : !torch.int, !torch.int -> !torch.int
    %57 = torch.aten.sub.int %54, %56 : !torch.int, !torch.int -> !torch.int
    %58 = torch.aten.sub.int %57, %int1 : !torch.int, !torch.int -> !torch.int
    %59 = torch.prim.If %arg5 -> (!torch.int) {
      %86 = torch.aten.sub.int %16, %int1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %86 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %60 = torch.aten.add.int %58, %59 : !torch.int, !torch.int -> !torch.int
    %61 = torch.aten.floordiv.int %60, %16 : !torch.int, !torch.int -> !torch.int
    %62 = torch.aten.add.int %61, %int1 : !torch.int, !torch.int -> !torch.int
    %63 = torch.prim.If %arg5 -> (!torch.int) {
      %86 = torch.aten.mul.int %61, %16 : !torch.int, !torch.int -> !torch.int
      %87 = torch.aten.add.int %39, %23 : !torch.int, !torch.int -> !torch.int
      %88 = torch.aten.ge.int %86, %87 : !torch.int, !torch.int -> !torch.bool
      %89 = torch.prim.If %88 -> (!torch.int) {
        torch.prim.If.yield %61 : !torch.int
      } else {
        torch.prim.If.yield %62 : !torch.int
      }
      torch.prim.If.yield %89 : !torch.int
    } else {
      torch.prim.If.yield %62 : !torch.int
    }
    %64 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %65 = torch.aten.gt.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
    %66 = torch.prim.If %65 -> (!torch.bool) {
      %86 = torch.aten.gt.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %86 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %66 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %67 = torch.aten.gt.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    %68 = torch.prim.If %67 -> (!torch.bool) {
      %86 = torch.aten.gt.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %86 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %68 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %69 = torch.aten.gt.int %27, %int0 : !torch.int, !torch.int -> !torch.bool
    %70 = torch.prim.If %69 -> (!torch.bool) {
      %86 = torch.aten.gt.int %30, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %86 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %70 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %71 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %72 = torch.aten.ne.int %71, %int0 : !torch.int, !torch.int -> !torch.bool
    %73 = torch.prim.If %72 -> (!torch.bool) {
      %86 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
      %87 = torch.aten.ne.int %86, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %74 = torch.aten.eq.int %64, %int3 : !torch.int, !torch.int -> !torch.bool
    %75 = torch.prim.If %74 -> (!torch.bool) {
      %86 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %87 = torch.aten.ne.int %86, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %76 = torch.prim.If %75 -> (!torch.bool) {
      torch.prim.If.yield %73 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %77 = torch.prim.If %76 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %86 = torch.aten.eq.int %64, %int4 : !torch.int, !torch.int -> !torch.bool
      %87 = torch.prim.If %86 -> (!torch.bool) {
        torch.prim.If.yield %73 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %88 = torch.prim.If %87 -> (!torch.bool) {
        %89 = torch.aten.__getitem__.t %arg0, %int3 : !torch.list<int>, !torch.int -> !torch.int
        %90 = torch.aten.ne.int %89, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %90 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %77 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %78 = torch.aten.floordiv.int %6, %int2 : !torch.int, !torch.int -> !torch.int
    %79 = torch.aten.ge.int %78, %23 : !torch.int, !torch.int -> !torch.bool
    %80 = torch.prim.If %79 -> (!torch.bool) {
      %86 = torch.aten.floordiv.int %3, %int2 : !torch.int, !torch.int -> !torch.int
      %87 = torch.aten.ge.int %86, %20 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %80 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %81 = torch.aten.ge.int %63, %int1 : !torch.int, !torch.int -> !torch.bool
    %82 = torch.prim.If %81 -> (!torch.bool) {
      %86 = torch.aten.ge.int %51, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %86 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %82 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %83 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %84 = torch.aten.eq.int %83, %int3 : !torch.int, !torch.int -> !torch.bool
    %85 = torch.prim.If %84 -> (!torch.list<int>) {
      %86 = torch.prim.ListConstruct %37, %51, %63 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %86 : !torch.list<int>
    } else {
      %86 = torch.prim.ListConstruct %36, %37, %51, %63 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %86 : !torch.list<int>
    }
    return %85 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.pooling_output_shape(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.bool) -> !torch.int {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: stride should not be zeero"
    %int0 = torch.constant.int 0
    %0 = torch.aten.ne.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %0 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %1 = call @__torch__.torch.jit._shape_functions.pooling_output_shape_pad_lr(%arg0, %arg1, %arg2, %arg2, %arg3, %arg4, %arg5) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    return %1 : !torch.int
  }
  func.func @__torch__.torch.jit._shape_functions.pooling_output_shape_pad_lr(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.bool) -> !torch.int {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
    %8 = call @__torch__.torch.jit._shape_functions.div_rtn(%7, %arg4) : (!torch.int, !torch.int) -> !torch.int
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
  func.func @__torch__.torch.jit._shape_functions.div_rtn(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
    %0 = torch.aten.floordiv.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
    return %0 : !torch.int
  }
  func.func @__torch__.torch.jit._shape_functions.pool2d_shape_check(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.int, %arg7: !torch.int, %arg8: !torch.int, %arg9: !torch.int, %arg10: !torch.int, %arg11: !torch.int, %arg12: !torch.int, %arg13: !torch.int) -> !torch.none {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
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
  func.func @__torch__.torch.jit._shape_functions.max_pool2d_with_indices(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %false = torch.constant.bool false
    %str = torch.constant.str "AssertionError: stride should not be zeero"
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %str_0 = torch.constant.str "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    %str_2 = torch.constant.str "AssertionError: max_pool2d: padding must be either be a single int, or a tuple of two ints"
    %str_3 = torch.constant.str "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
    %str_4 = torch.constant.str "AssertionError: "
    %int-4 = torch.constant.int -4
    %int-3 = torch.constant.int -3
    %int-2 = torch.constant.int -2
    %int-1 = torch.constant.int -1
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %5 = torch.aten.eq.int %4, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %87 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    }
    %7 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    %10 = torch.prim.If %9 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %10 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %11 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %87 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    }
    %14 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %15 = torch.aten.eq.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %6 : !torch.int
    } else {
      %87 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int1 : !torch.int, !torch.int -> !torch.bool
      %89 = torch.prim.If %88 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        %90 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %90 : !torch.int
      }
      torch.prim.If.yield %89 : !torch.int
    }
    %17 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %19 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %20 = torch.aten.__getitem__.t %arg3, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %21 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %20 : !torch.int
    } else {
      %87 = torch.aten.__getitem__.t %arg3, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    }
    %24 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %25 = torch.aten.eq.int %24, %int1 : !torch.int, !torch.int -> !torch.bool
    %26 = torch.prim.If %25 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %26 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_3, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %27 = torch.aten.__getitem__.t %arg4, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %28 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %29 = torch.aten.eq.int %28, %int1 : !torch.int, !torch.int -> !torch.bool
    %30 = torch.prim.If %29 -> (!torch.int) {
      torch.prim.If.yield %27 : !torch.int
    } else {
      %87 = torch.aten.__getitem__.t %arg4, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    }
    %31 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %32 = torch.aten.eq.int %31, %int3 : !torch.int, !torch.int -> !torch.bool
    %33 = torch.prim.If %32 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %88 = torch.aten.eq.int %87, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    }
    torch.prim.If %33 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %34 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %35 = torch.aten.eq.int %34, %int4 : !torch.int, !torch.int -> !torch.bool
    %36 = torch.prim.If %35 -> (!torch.int) {
      %87 = torch.aten.__getitem__.t %arg0, %int-4 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    } else {
      torch.prim.If.yield %int1 : !torch.int
    }
    %37 = torch.aten.__getitem__.t %arg0, %int-3 : !torch.list<int>, !torch.int -> !torch.int
    %38 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
    %39 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
    %40 = torch.aten.ne.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %40 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %41 = torch.aten.add.int %38, %20 : !torch.int, !torch.int -> !torch.int
    %42 = torch.aten.add.int %41, %20 : !torch.int, !torch.int -> !torch.int
    %43 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
    %44 = torch.aten.mul.int %27, %43 : !torch.int, !torch.int -> !torch.int
    %45 = torch.aten.sub.int %42, %44 : !torch.int, !torch.int -> !torch.int
    %46 = torch.aten.sub.int %45, %int1 : !torch.int, !torch.int -> !torch.int
    %47 = torch.prim.If %arg5 -> (!torch.int) {
      %87 = torch.aten.sub.int %13, %int1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %48 = torch.aten.add.int %46, %47 : !torch.int, !torch.int -> !torch.int
    %49 = torch.aten.floordiv.int %48, %13 : !torch.int, !torch.int -> !torch.int
    %50 = torch.aten.add.int %49, %int1 : !torch.int, !torch.int -> !torch.int
    %51 = torch.prim.If %arg5 -> (!torch.int) {
      %87 = torch.aten.mul.int %49, %13 : !torch.int, !torch.int -> !torch.int
      %88 = torch.aten.add.int %38, %20 : !torch.int, !torch.int -> !torch.int
      %89 = torch.aten.ge.int %87, %88 : !torch.int, !torch.int -> !torch.bool
      %90 = torch.prim.If %89 -> (!torch.int) {
        torch.prim.If.yield %49 : !torch.int
      } else {
        torch.prim.If.yield %50 : !torch.int
      }
      torch.prim.If.yield %90 : !torch.int
    } else {
      torch.prim.If.yield %50 : !torch.int
    }
    %52 = torch.aten.ne.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %52 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %53 = torch.aten.add.int %39, %23 : !torch.int, !torch.int -> !torch.int
    %54 = torch.aten.add.int %53, %23 : !torch.int, !torch.int -> !torch.int
    %55 = torch.aten.sub.int %6, %int1 : !torch.int, !torch.int -> !torch.int
    %56 = torch.aten.mul.int %30, %55 : !torch.int, !torch.int -> !torch.int
    %57 = torch.aten.sub.int %54, %56 : !torch.int, !torch.int -> !torch.int
    %58 = torch.aten.sub.int %57, %int1 : !torch.int, !torch.int -> !torch.int
    %59 = torch.prim.If %arg5 -> (!torch.int) {
      %87 = torch.aten.sub.int %16, %int1 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %87 : !torch.int
    } else {
      torch.prim.If.yield %int0 : !torch.int
    }
    %60 = torch.aten.add.int %58, %59 : !torch.int, !torch.int -> !torch.int
    %61 = torch.aten.floordiv.int %60, %16 : !torch.int, !torch.int -> !torch.int
    %62 = torch.aten.add.int %61, %int1 : !torch.int, !torch.int -> !torch.int
    %63 = torch.prim.If %arg5 -> (!torch.int) {
      %87 = torch.aten.mul.int %61, %16 : !torch.int, !torch.int -> !torch.int
      %88 = torch.aten.add.int %39, %23 : !torch.int, !torch.int -> !torch.int
      %89 = torch.aten.ge.int %87, %88 : !torch.int, !torch.int -> !torch.bool
      %90 = torch.prim.If %89 -> (!torch.int) {
        torch.prim.If.yield %61 : !torch.int
      } else {
        torch.prim.If.yield %62 : !torch.int
      }
      torch.prim.If.yield %90 : !torch.int
    } else {
      torch.prim.If.yield %62 : !torch.int
    }
    %64 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %65 = torch.aten.gt.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
    %66 = torch.prim.If %65 -> (!torch.bool) {
      %87 = torch.aten.gt.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %66 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %67 = torch.aten.gt.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    %68 = torch.prim.If %67 -> (!torch.bool) {
      %87 = torch.aten.gt.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %68 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %69 = torch.aten.gt.int %27, %int0 : !torch.int, !torch.int -> !torch.bool
    %70 = torch.prim.If %69 -> (!torch.bool) {
      %87 = torch.aten.gt.int %30, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %70 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %71 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %72 = torch.aten.ne.int %71, %int0 : !torch.int, !torch.int -> !torch.bool
    %73 = torch.prim.If %72 -> (!torch.bool) {
      %87 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
      %88 = torch.aten.ne.int %87, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %74 = torch.aten.eq.int %64, %int3 : !torch.int, !torch.int -> !torch.bool
    %75 = torch.prim.If %74 -> (!torch.bool) {
      %87 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %88 = torch.aten.ne.int %87, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %76 = torch.prim.If %75 -> (!torch.bool) {
      torch.prim.If.yield %73 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %77 = torch.prim.If %76 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %87 = torch.aten.eq.int %64, %int4 : !torch.int, !torch.int -> !torch.bool
      %88 = torch.prim.If %87 -> (!torch.bool) {
        torch.prim.If.yield %73 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %89 = torch.prim.If %88 -> (!torch.bool) {
        %90 = torch.aten.__getitem__.t %arg0, %int3 : !torch.list<int>, !torch.int -> !torch.int
        %91 = torch.aten.ne.int %90, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %91 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %89 : !torch.bool
    }
    torch.prim.If %77 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %78 = torch.aten.floordiv.int %6, %int2 : !torch.int, !torch.int -> !torch.int
    %79 = torch.aten.ge.int %78, %23 : !torch.int, !torch.int -> !torch.bool
    %80 = torch.prim.If %79 -> (!torch.bool) {
      %87 = torch.aten.floordiv.int %3, %int2 : !torch.int, !torch.int -> !torch.int
      %88 = torch.aten.ge.int %87, %20 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %88 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %80 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %81 = torch.aten.ge.int %63, %int1 : !torch.int, !torch.int -> !torch.bool
    %82 = torch.prim.If %81 -> (!torch.bool) {
      %87 = torch.aten.ge.int %51, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %87 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    torch.prim.If %82 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_4, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %83 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %84 = torch.aten.eq.int %83, %int3 : !torch.int, !torch.int -> !torch.bool
    %85 = torch.prim.If %84 -> (!torch.list<int>) {
      %87 = torch.prim.ListConstruct %37, %51, %63 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %87 : !torch.list<int>
    } else {
      %87 = torch.prim.ListConstruct %36, %37, %51, %63 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %87 : !torch.list<int>
    }
    %86 = torch.prim.TupleConstruct %85, %85 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %86 : !torch.tuple<list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.transpose(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %3 = torch.aten.neg.int %2 : !torch.int -> !torch.int
    %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.lt.int %arg1, %3 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %21 = torch.aten.gt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %21 : !torch.bool
    }
    %7 = torch.aten.__not__ %6 : !torch.bool -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.int) {
      %21 = torch.aten.add.int %arg1, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %21 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %10 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %11 = torch.prim.If %10 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %12 = torch.aten.neg.int %11 : !torch.int -> !torch.int
    %13 = torch.aten.sub.int %11, %int1 : !torch.int, !torch.int -> !torch.int
    %14 = torch.aten.lt.int %arg2, %12 : !torch.int, !torch.int -> !torch.bool
    %15 = torch.prim.If %14 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %21 = torch.aten.gt.int %arg2, %13 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %21 : !torch.bool
    }
    %16 = torch.aten.__not__ %15 : !torch.bool -> !torch.bool
    torch.prim.If %16 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %17 = torch.aten.lt.int %arg2, %int0 : !torch.int, !torch.int -> !torch.bool
    %18 = torch.prim.If %17 -> (!torch.int) {
      %21 = torch.aten.add.int %arg2, %11 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %21 : !torch.int
    } else {
      torch.prim.If.yield %arg2 : !torch.int
    }
    %19 = torch.aten.eq.int %9, %18 : !torch.int, !torch.int -> !torch.bool
    %20 = torch.prim.If %19 -> (!torch.list<int>) {
      %21 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %22 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      torch.prim.Loop %22, %true, init() {
      ^bb0(%arg3: !torch.int):
        %23 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %24 = torch.aten.append.t %21, %23 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %21 : !torch.list<int>
    } else {
      %21 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %0, %true, init() {
      ^bb0(%arg3: !torch.int):
        %22 = torch.aten.eq.int %arg3, %9 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %22 -> () {
          %23 = torch.aten.__getitem__.t %arg0, %18 : !torch.list<int>, !torch.int -> !torch.int
          %24 = torch.aten.append.t %21, %23 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          %23 = torch.aten.eq.int %arg3, %18 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %23 -> () {
            %24 = torch.aten.__getitem__.t %arg0, %9 : !torch.list<int>, !torch.int -> !torch.int
            %25 = torch.aten.append.t %21, %24 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.If.yield
          } else {
            %24 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
            %25 = torch.aten.append.t %21, %24 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.If.yield
          }
          torch.prim.If.yield
        }
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %21 : !torch.list<int>
    }
    return %20 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.conv1d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int3 = torch.constant.int 3
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %5 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %6 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %7 = torch.prim.Loop %6, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg4, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %10 = torch.prim.Loop %9, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg3, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %11 = torch.aten.__not__ %10 : !torch.bool -> !torch.bool
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.eq.int %5, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %12 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %13 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %14 = torch.aten.ge.int %13, %arg6 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %14 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %15 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %16 = torch.aten.remainder.int %15, %arg6 : !torch.int, !torch.int -> !torch.int
    %17 = torch.aten.eq.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %17 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %18 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %19 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %20 = torch.aten.mul.int %19, %arg6 : !torch.int, !torch.int -> !torch.int
    %21 = torch.aten.eq.int %18, %20 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %21 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %22 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %34 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %35 = torch.aten.len.t %34 : !torch.list<int> -> !torch.int
      %36 = torch.aten.eq.int %35, %int1 : !torch.int, !torch.int -> !torch.bool
      %37 = torch.prim.If %36 -> (!torch.bool) {
        %38 = torch.aten.__getitem__.t %34, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %39 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %40 = torch.aten.eq.int %38, %39 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %40 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %37 : !torch.bool
    }
    torch.prim.If %23 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %24 = torch.aten.__range_length %int2, %4, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %24, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %36 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %37 = torch.aten.__getitem__.t %arg4, %36 : !torch.list<int>, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %37, %int2 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %35, %38 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %41 = torch.aten.__getitem__.t %arg5, %40 : !torch.list<int>, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.sub.int %42, %int1 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.mul.int %41, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.add.int %44, %int1 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.ge.int %39, %45 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %46 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %25 = torch.aten.len.t %arg5 : !torch.list<int> -> !torch.int
    %26 = torch.aten.gt.int %25, %int0 : !torch.int, !torch.int -> !torch.bool
    %27 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %28 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %29 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %30 = torch.aten.append.t %28, %29 : !torch.list<int>, !torch.int -> !torch.list<int>
    %31 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %32 = torch.aten.append.t %28, %31 : !torch.list<int>, !torch.int -> !torch.list<int>
    %33 = torch.aten.__range_length %int2, %27, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %33, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.prim.If %26 -> (!torch.int) {
        %51 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
        %52 = torch.aten.__getitem__.t %arg5, %51 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %52 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %36 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %37 = torch.aten.sub.int %36, %int1 : !torch.int, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %35, %37 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %38, %int1 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %41 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg4, %41 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.mul.int %42, %int2 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.add.int %40, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.sub.int %44, %39 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %47 = torch.aten.__getitem__.t %arg3, %46 : !torch.list<int>, !torch.int -> !torch.int
      %48 = torch.aten.floordiv.int %45, %47 : !torch.int, !torch.int -> !torch.int
      %49 = torch.aten.add.int %48, %int1 : !torch.int, !torch.int -> !torch.int
      %50 = torch.aten.append.t %28, %49 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %28 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.conv_output_size(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = call @__torch__.torch.jit._shape_functions.check_shape_forward(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.none
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
  func.func @__torch__.torch.jit._shape_functions.check_shape_forward(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.none {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = call @__torch__.torch.jit._shape_functions.check_non_negative(%arg4) : (!torch.list<int>) -> !torch.bool
    %3 = torch.aten.__not__ %2 : !torch.bool -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = call @__torch__.torch.jit._shape_functions.check_non_negative(%arg3) : (!torch.list<int>) -> !torch.bool
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
  func.func @__torch__.torch.jit._shape_functions.check_non_negative(%arg0: !torch.list<int>) -> !torch.bool {
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
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
  func.func @__torch__.torch.jit._shape_functions.conv2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int4 = torch.constant.int 4
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
    %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %5 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %6 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %7 = torch.prim.Loop %6, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg4, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %10 = torch.prim.Loop %9, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg3, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %11 = torch.aten.__not__ %10 : !torch.bool -> !torch.bool
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.eq.int %5, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %12 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %13 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %14 = torch.aten.ge.int %13, %arg6 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %14 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %15 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %16 = torch.aten.remainder.int %15, %arg6 : !torch.int, !torch.int -> !torch.int
    %17 = torch.aten.eq.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %17 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %18 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %19 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %20 = torch.aten.mul.int %19, %arg6 : !torch.int, !torch.int -> !torch.int
    %21 = torch.aten.eq.int %18, %20 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %21 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %22 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %34 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %35 = torch.aten.len.t %34 : !torch.list<int> -> !torch.int
      %36 = torch.aten.eq.int %35, %int1 : !torch.int, !torch.int -> !torch.bool
      %37 = torch.prim.If %36 -> (!torch.bool) {
        %38 = torch.aten.__getitem__.t %34, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %39 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %40 = torch.aten.eq.int %38, %39 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %40 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %37 : !torch.bool
    }
    torch.prim.If %23 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %24 = torch.aten.__range_length %int2, %4, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %24, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %36 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %37 = torch.aten.__getitem__.t %arg4, %36 : !torch.list<int>, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %37, %int2 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %35, %38 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %41 = torch.aten.__getitem__.t %arg5, %40 : !torch.list<int>, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.sub.int %42, %int1 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.mul.int %41, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.add.int %44, %int1 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.ge.int %39, %45 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %46 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %25 = torch.aten.len.t %arg5 : !torch.list<int> -> !torch.int
    %26 = torch.aten.gt.int %25, %int0 : !torch.int, !torch.int -> !torch.bool
    %27 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %28 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %29 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %30 = torch.aten.append.t %28, %29 : !torch.list<int>, !torch.int -> !torch.list<int>
    %31 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %32 = torch.aten.append.t %28, %31 : !torch.list<int>, !torch.int -> !torch.list<int>
    %33 = torch.aten.__range_length %int2, %27, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %33, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.prim.If %26 -> (!torch.int) {
        %51 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
        %52 = torch.aten.__getitem__.t %arg5, %51 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %52 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %36 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %37 = torch.aten.sub.int %36, %int1 : !torch.int, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %35, %37 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %38, %int1 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %41 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg4, %41 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.mul.int %42, %int2 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.add.int %40, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.sub.int %44, %39 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %47 = torch.aten.__getitem__.t %arg3, %46 : !torch.list<int>, !torch.int -> !torch.int
      %48 = torch.aten.floordiv.int %45, %47 : !torch.int, !torch.int -> !torch.int
      %49 = torch.aten.add.int %48, %int1 : !torch.int, !torch.int -> !torch.int
      %50 = torch.aten.append.t %28, %49 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %28 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.batch_norm(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool) -> !torch.list<int> {
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg9: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg9 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.append.t %0, %2 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.conv3d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int5 = torch.constant.int 5
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %5 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %6 = torch.aten.len.t %arg4 : !torch.list<int> -> !torch.int
    %7 = torch.prim.Loop %6, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg4, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %9 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %10 = torch.prim.Loop %9, %true, init(%false) {
    ^bb0(%arg7: !torch.int, %arg8: !torch.bool):
      %34 = torch.aten.__getitem__.t %arg3, %arg7 : !torch.list<int>, !torch.int -> !torch.int
      %35 = torch.aten.lt.int %34, %int0 : !torch.int, !torch.int -> !torch.bool
      %36 = torch.prim.If %35 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        torch.prim.If.yield %arg8 : !torch.bool
      }
      torch.prim.Loop.condition %true, iter(%36 : !torch.bool)
    } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
    %11 = torch.aten.__not__ %10 : !torch.bool -> !torch.bool
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.eq.int %5, %4 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %12 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %13 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %14 = torch.aten.ge.int %13, %arg6 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %14 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %15 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %16 = torch.aten.remainder.int %15, %arg6 : !torch.int, !torch.int -> !torch.int
    %17 = torch.aten.eq.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %17 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %18 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %19 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %20 = torch.aten.mul.int %19, %arg6 : !torch.int, !torch.int -> !torch.int
    %21 = torch.aten.eq.int %18, %20 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %21 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %22 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %34 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %35 = torch.aten.len.t %34 : !torch.list<int> -> !torch.int
      %36 = torch.aten.eq.int %35, %int1 : !torch.int, !torch.int -> !torch.bool
      %37 = torch.prim.If %36 -> (!torch.bool) {
        %38 = torch.aten.__getitem__.t %34, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %39 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %40 = torch.aten.eq.int %38, %39 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %40 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %37 : !torch.bool
    }
    torch.prim.If %23 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %24 = torch.aten.__range_length %int2, %4, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %24, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %36 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %37 = torch.aten.__getitem__.t %arg4, %36 : !torch.list<int>, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %37, %int2 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %35, %38 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %41 = torch.aten.__getitem__.t %arg5, %40 : !torch.list<int>, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.sub.int %42, %int1 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.mul.int %41, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.add.int %44, %int1 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.ge.int %39, %45 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %46 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %25 = torch.aten.len.t %arg5 : !torch.list<int> -> !torch.int
    %26 = torch.aten.gt.int %25, %int0 : !torch.int, !torch.int -> !torch.bool
    %27 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %28 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %29 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %30 = torch.aten.append.t %28, %29 : !torch.list<int>, !torch.int -> !torch.list<int>
    %31 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %32 = torch.aten.append.t %28, %31 : !torch.list<int>, !torch.int -> !torch.list<int>
    %33 = torch.aten.__range_length %int2, %27, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %33, %true, init() {
    ^bb0(%arg7: !torch.int):
      %34 = torch.aten.__derive_index %arg7, %int2, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
      %35 = torch.prim.If %26 -> (!torch.int) {
        %51 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
        %52 = torch.aten.__getitem__.t %arg5, %51 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %52 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %36 = torch.aten.__getitem__.t %arg1, %34 : !torch.list<int>, !torch.int -> !torch.int
      %37 = torch.aten.sub.int %36, %int1 : !torch.int, !torch.int -> !torch.int
      %38 = torch.aten.mul.int %35, %37 : !torch.int, !torch.int -> !torch.int
      %39 = torch.aten.add.int %38, %int1 : !torch.int, !torch.int -> !torch.int
      %40 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
      %41 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %42 = torch.aten.__getitem__.t %arg4, %41 : !torch.list<int>, !torch.int -> !torch.int
      %43 = torch.aten.mul.int %42, %int2 : !torch.int, !torch.int -> !torch.int
      %44 = torch.aten.add.int %40, %43 : !torch.int, !torch.int -> !torch.int
      %45 = torch.aten.sub.int %44, %39 : !torch.int, !torch.int -> !torch.int
      %46 = torch.aten.sub.int %34, %int2 : !torch.int, !torch.int -> !torch.int
      %47 = torch.aten.__getitem__.t %arg3, %46 : !torch.list<int>, !torch.int -> !torch.int
      %48 = torch.aten.floordiv.int %45, %47 : !torch.int, !torch.int -> !torch.int
      %49 = torch.aten.add.int %48, %int1 : !torch.int, !torch.int -> !torch.int
      %50 = torch.aten.append.t %28, %49 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %28 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.conv_backwards(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.optional<list<int>>) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg4: !torch.int):
      %7 = torch.aten.__getitem__.t %arg1, %arg4 : !torch.list<int>, !torch.int -> !torch.int
      %8 = torch.aten.append.t %0, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %3 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    torch.prim.Loop %3, %true, init() {
    ^bb0(%arg4: !torch.int):
      %7 = torch.aten.__getitem__.t %arg2, %arg4 : !torch.list<int>, !torch.int -> !torch.int
      %8 = torch.aten.append.t %2, %7 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %4 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.prim.ListConstruct %4 : (!torch.int) -> !torch.list<int>
    %6 = torch.prim.TupleConstruct %0, %2, %5 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
    return %6 : !torch.tuple<list<int>, list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.flatten(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %0 : !torch.int
    }
    %3 = torch.aten.neg.int %2 : !torch.int -> !torch.int
    %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.lt.int %arg1, %3 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %24 = torch.aten.gt.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %24 : !torch.bool
    }
    %7 = torch.aten.__not__ %6 : !torch.bool -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.int) {
      %24 = torch.aten.add.int %arg1, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %24 : !torch.int
    } else {
      torch.prim.If.yield %arg1 : !torch.int
    }
    %10 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %11 = torch.aten.le.int %10, %int0 : !torch.int, !torch.int -> !torch.bool
    %12 = torch.prim.If %11 -> (!torch.int) {
      torch.prim.If.yield %int1 : !torch.int
    } else {
      torch.prim.If.yield %10 : !torch.int
    }
    %13 = torch.aten.neg.int %12 : !torch.int -> !torch.int
    %14 = torch.aten.sub.int %12, %int1 : !torch.int, !torch.int -> !torch.int
    %15 = torch.aten.lt.int %arg2, %13 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %24 = torch.aten.gt.int %arg2, %14 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %24 : !torch.bool
    }
    %17 = torch.aten.__not__ %16 : !torch.bool -> !torch.bool
    torch.prim.If %17 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %18 = torch.aten.lt.int %arg2, %int0 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.int) {
      %24 = torch.aten.add.int %arg2, %12 : !torch.int, !torch.int -> !torch.int
      torch.prim.If.yield %24 : !torch.int
    } else {
      torch.prim.If.yield %arg2 : !torch.int
    }
    %20 = torch.aten.le.int %9, %19 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %20 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %21 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int0 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.list<int>) {
      %24 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %24 : !torch.list<int>
    } else {
      %24 = torch.aten.eq.int %9, %19 : !torch.int, !torch.int -> !torch.bool
      %25 = torch.prim.If %24 -> (!torch.list<int>) {
        %26 = torch.prim.ListConstruct  : () -> !torch.list<int>
        %27 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        torch.prim.Loop %27, %true, init() {
        ^bb0(%arg3: !torch.int):
          %28 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
          %29 = torch.aten.append.t %26, %28 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %26 : !torch.list<int>
      } else {
        %26 = torch.aten.add.int %19, %int1 : !torch.int, !torch.int -> !torch.int
        %27 = torch.aten.__range_length %9, %26, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        %28 = torch.prim.Loop %27, %true, init(%int1) {
        ^bb0(%arg3: !torch.int, %arg4: !torch.int):
          %34 = torch.aten.__derive_index %arg3, %9, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %35 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
          %36 = torch.aten.mul.int %arg4, %35 : !torch.int, !torch.int -> !torch.int
          torch.prim.Loop.condition %true, iter(%36 : !torch.int)
        } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
        %29 = torch.prim.ListConstruct  : () -> !torch.list<int>
        torch.prim.Loop %9, %true, init() {
        ^bb0(%arg3: !torch.int):
          %34 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
          %35 = torch.aten.append.t %29, %34 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        %30 = torch.aten.append.t %29, %28 : !torch.list<int>, !torch.int -> !torch.list<int>
        %31 = torch.aten.add.int %19, %int1 : !torch.int, !torch.int -> !torch.int
        %32 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %33 = torch.aten.__range_length %31, %32, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        torch.prim.Loop %33, %true, init() {
        ^bb0(%arg3: !torch.int):
          %34 = torch.aten.__derive_index %arg3, %31, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %35 = torch.aten.__getitem__.t %arg0, %34 : !torch.list<int>, !torch.int -> !torch.int
          %36 = torch.aten.append.t %29, %35 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %29 : !torch.list<int>
      }
      torch.prim.If.yield %25 : !torch.list<int>
    }
    return %23 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.cat(%arg0: !torch.list<list<int>>, %arg1: !torch.int) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: Sizes of tensors must match except in dimension"
    %str_0 = torch.constant.str "AssertionError: Tensors must have same number of dimensions"
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    torch.prim.Loop %0, %true, init() {
    ^bb0(%arg2: !torch.int):
      %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %14 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
      %15 = torch.aten.gt.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %15 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %1 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    %2 = torch.derefine %none : !torch.none to !torch.optional<int>
    %3 = torch.prim.Loop %1, %true, init(%2) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.optional<int>):
      %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %14 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
      %15 = torch.aten.eq.int %14, %int1 : !torch.int, !torch.int -> !torch.bool
      %16 = torch.prim.If %15 -> (!torch.bool) {
        %19 = torch.aten.__getitem__.t %13, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.eq.int %19, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %17 = torch.aten.__not__ %16 : !torch.bool -> !torch.bool
      %18 = torch.prim.If %17 -> (!torch.optional<int>) {
        %19 = torch.aten.__is__ %arg3, %none : !torch.optional<int>, !torch.none -> !torch.bool
        %20 = torch.prim.If %19 -> (!torch.int) {
          %22 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
          %23 = torch.aten.le.int %22, %int0 : !torch.int, !torch.int -> !torch.bool
          %24 = torch.prim.If %23 -> (!torch.int) {
            torch.prim.If.yield %int1 : !torch.int
          } else {
            torch.prim.If.yield %22 : !torch.int
          }
          %25 = torch.aten.neg.int %24 : !torch.int -> !torch.int
          %26 = torch.aten.sub.int %24, %int1 : !torch.int, !torch.int -> !torch.int
          %27 = torch.aten.lt.int %arg1, %25 : !torch.int, !torch.int -> !torch.bool
          %28 = torch.prim.If %27 -> (!torch.bool) {
            torch.prim.If.yield %true : !torch.bool
          } else {
            %32 = torch.aten.gt.int %arg1, %26 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If.yield %32 : !torch.bool
          }
          %29 = torch.aten.__not__ %28 : !torch.bool -> !torch.bool
          torch.prim.If %29 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %30 = torch.aten.lt.int %arg1, %int0 : !torch.int, !torch.int -> !torch.bool
          %31 = torch.prim.If %30 -> (!torch.int) {
            %32 = torch.aten.add.int %arg1, %24 : !torch.int, !torch.int -> !torch.int
            torch.prim.If.yield %32 : !torch.int
          } else {
            torch.prim.If.yield %arg1 : !torch.int
          }
          torch.prim.If.yield %31 : !torch.int
        } else {
          %22 = torch.prim.unchecked_cast %arg3 : !torch.optional<int> -> !torch.int
          torch.prim.If.yield %22 : !torch.int
        }
        %21 = torch.derefine %20 : !torch.int to !torch.optional<int>
        torch.prim.If.yield %21 : !torch.optional<int>
      } else {
        torch.prim.If.yield %arg3 : !torch.optional<int>
      }
      torch.prim.Loop.condition %true, iter(%18 : !torch.optional<int>)
    } : (!torch.int, !torch.bool, !torch.optional<int>) -> !torch.optional<int>
    %4 = torch.aten.__is__ %3, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %5 = torch.prim.If %4 -> (!torch.int) {
      torch.prim.If.yield %arg1 : !torch.int
    } else {
      %13 = torch.prim.unchecked_cast %3 : !torch.optional<int> -> !torch.int
      torch.prim.If.yield %13 : !torch.int
    }
    %6 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    %7 = torch.aten.gt.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %7 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %8 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
    %9 = torch.derefine %none : !torch.none to !torch.optional<list<int>>
    %10 = torch.prim.Loop %8, %true, init(%9) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.optional<list<int>>):
      %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %14 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
      %15 = torch.prim.Loop %14, %true, init(%int1) {
      ^bb0(%arg4: !torch.int, %arg5: !torch.int):
        %20 = torch.aten.__getitem__.t %13, %arg4 : !torch.list<int>, !torch.int -> !torch.int
        %21 = torch.aten.mul.int %arg5, %20 : !torch.int, !torch.int -> !torch.int
        torch.prim.Loop.condition %true, iter(%21 : !torch.int)
      } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
      %16 = torch.aten.eq.int %15, %int0 : !torch.int, !torch.int -> !torch.bool
      %17 = torch.prim.If %16 -> (!torch.bool) {
        %20 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
        %21 = torch.aten.eq.int %20, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %21 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %18 = torch.aten.__not__ %17 : !torch.bool -> !torch.bool
      %19 = torch.prim.If %18 -> (!torch.optional<list<int>>) {
        %20 = torch.derefine %13 : !torch.list<int> to !torch.optional<list<int>>
        torch.prim.If.yield %20 : !torch.optional<list<int>>
      } else {
        torch.prim.If.yield %arg3 : !torch.optional<list<int>>
      }
      torch.prim.Loop.condition %true, iter(%19 : !torch.optional<list<int>>)
    } : (!torch.int, !torch.bool, !torch.optional<list<int>>) -> !torch.optional<list<int>>
    %11 = torch.aten.__is__ %10, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %12 = torch.prim.If %11 -> (!torch.list<int>) {
      %13 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %13 : !torch.list<int>
    } else {
      %13 = torch.prim.unchecked_cast %10 : !torch.optional<list<int>> -> !torch.list<int>
      %14 = torch.aten.len.t %arg0 : !torch.list<list<int>> -> !torch.int
      %15 = torch.prim.Loop %14, %true, init(%int0) {
      ^bb0(%arg2: !torch.int, %arg3: !torch.int):
        %19 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
        %20 = torch.aten.len.t %19 : !torch.list<int> -> !torch.int
        %21 = torch.prim.Loop %20, %true, init(%int1) {
        ^bb0(%arg4: !torch.int, %arg5: !torch.int):
          %26 = torch.aten.__getitem__.t %19, %arg4 : !torch.list<int>, !torch.int -> !torch.int
          %27 = torch.aten.mul.int %arg5, %26 : !torch.int, !torch.int -> !torch.int
          torch.prim.Loop.condition %true, iter(%27 : !torch.int)
        } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
        %22 = torch.aten.eq.int %21, %int0 : !torch.int, !torch.int -> !torch.bool
        %23 = torch.prim.If %22 -> (!torch.bool) {
          %26 = torch.aten.len.t %19 : !torch.list<int> -> !torch.int
          %27 = torch.aten.eq.int %26, %int1 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %27 : !torch.bool
        } else {
          torch.prim.If.yield %false : !torch.bool
        }
        %24 = torch.aten.__not__ %23 : !torch.bool -> !torch.bool
        %25 = torch.prim.If %24 -> (!torch.int) {
          %26 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
          %27 = torch.aten.len.t %19 : !torch.list<int> -> !torch.int
          %28 = torch.aten.eq.int %26, %27 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If %28 -> () {
            torch.prim.If.yield
          } else {
            torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
            torch.prim.If.yield
          }
          %29 = torch.aten.__range_length %int0, %26, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          torch.prim.Loop %29, %true, init() {
          ^bb0(%arg4: !torch.int):
            %32 = torch.aten.__derive_index %arg4, %int0, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
            %33 = torch.aten.ne.int %32, %5 : !torch.int, !torch.int -> !torch.bool
            torch.prim.If %33 -> () {
              %34 = torch.aten.__getitem__.t %13, %32 : !torch.list<int>, !torch.int -> !torch.int
              %35 = torch.aten.__getitem__.t %19, %32 : !torch.list<int>, !torch.int -> !torch.int
              %36 = torch.aten.eq.int %34, %35 : !torch.int, !torch.int -> !torch.bool
              torch.prim.If %36 -> () {
                torch.prim.If.yield
              } else {
                torch.prim.RaiseException %str, %none : !torch.str, !torch.none
                torch.prim.If.yield
              }
              torch.prim.If.yield
            } else {
              torch.prim.If.yield
            }
            torch.prim.Loop.condition %true, iter()
          } : (!torch.int, !torch.bool) -> ()
          %30 = torch.aten.__getitem__.t %19, %5 : !torch.list<int>, !torch.int -> !torch.int
          %31 = torch.aten.add.int %arg3, %30 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %31 : !torch.int
        } else {
          torch.prim.If.yield %arg3 : !torch.int
        }
        torch.prim.Loop.condition %true, iter(%25 : !torch.int)
      } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
      %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %17 = torch.aten.len.t %13 : !torch.list<int> -> !torch.int
      torch.prim.Loop %17, %true, init() {
      ^bb0(%arg2: !torch.int):
        %19 = torch.aten.__getitem__.t %13, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %20 = torch.aten.append.t %16, %19 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      %18 = torch.aten._set_item.t %16, %5, %15 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield %16 : !torch.list<int>
    }
    return %12 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.check_cat_no_zero_dim(%arg0: !torch.list<list<int>>) -> !torch.none {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
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
  func.func @__torch__.torch.jit._shape_functions.legacy_cat_wrap_dim(%arg0: !torch.int, %arg1: !torch.list<list<int>>) -> !torch.int {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg1 : !torch.list<list<int>> -> !torch.int
    %1 = torch.derefine %none : !torch.none to !torch.optional<int>
    %2 = torch.prim.Loop %0, %true, init(%1) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.optional<int>):
      %5 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<list<int>>, !torch.int -> !torch.list<int>
      %6 = torch.aten.len.t %5 : !torch.list<int> -> !torch.int
      %7 = torch.aten.eq.int %6, %int1 : !torch.int, !torch.int -> !torch.bool
      %8 = torch.prim.If %7 -> (!torch.bool) {
        %11 = torch.aten.__getitem__.t %5, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %12 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %9 = torch.aten.__not__ %8 : !torch.bool -> !torch.bool
      %10 = torch.prim.If %9 -> (!torch.optional<int>) {
        %11 = torch.aten.__is__ %arg3, %none : !torch.optional<int>, !torch.none -> !torch.bool
        %12 = torch.prim.If %11 -> (!torch.int) {
          %14 = torch.aten.len.t %5 : !torch.list<int> -> !torch.int
          %15 = func.call @__torch__.torch.jit._shape_functions.maybe_wrap_dim(%arg0, %14, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
          torch.prim.If.yield %15 : !torch.int
        } else {
          %14 = torch.prim.unchecked_cast %arg3 : !torch.optional<int> -> !torch.int
          torch.prim.If.yield %14 : !torch.int
        }
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
  func.func @__torch__.torch.jit._shape_functions.should_skip(%arg0: !torch.list<int>) -> !torch.bool {
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = call @__torch__.torch.jit._shape_functions.numel(%arg0) : (!torch.list<int>) -> !torch.int
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
  func.func @__torch__.torch.jit._shape_functions.numel(%arg0: !torch.list<int>) -> !torch.int {
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %3 = torch.aten.mul.int %arg2, %2 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%3 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    return %1 : !torch.int
  }
  func.func @__torch__.torch.jit._shape_functions.check_cat_shape_except_dim(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.none {
    %str = torch.constant.str "AssertionError: Sizes of tensors must match except in dimension"
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: Tensors must have same number of dimensions"
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.eq.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
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
          torch.prim.RaiseException %str, %none : !torch.str, !torch.none
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
  func.func @__torch__.torch.jit._shape_functions.permute(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int1 = torch.constant.int 1
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
      %8 = torch.aten.le.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.int) {
        torch.prim.If.yield %int1 : !torch.int
      } else {
        torch.prim.If.yield %3 : !torch.int
      }
      %10 = torch.aten.neg.int %9 : !torch.int -> !torch.int
      %11 = torch.aten.sub.int %9, %int1 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.lt.int %7, %10 : !torch.int, !torch.int -> !torch.bool
      %13 = torch.prim.If %12 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        %20 = torch.aten.gt.int %7, %11 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      }
      %14 = torch.aten.__not__ %13 : !torch.bool -> !torch.bool
      torch.prim.If %14 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %15 = torch.aten.lt.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
      %16 = torch.prim.If %15 -> (!torch.int) {
        %20 = torch.aten.add.int %7, %9 : !torch.int, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %7 : !torch.int
      }
      %17 = torch.aten.append.t %4, %16 : !torch.list<int>, !torch.int -> !torch.list<int>
      %18 = torch.aten.__getitem__.t %arg0, %16 : !torch.list<int>, !torch.int -> !torch.int
      %19 = torch.aten.append.t %5, %18 : !torch.list<int>, !torch.int -> !torch.list<int>
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
  func.func @__torch__.torch.jit._shape_functions.view(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: invalid shape"
    %false = torch.constant.bool false
    %str_0 = torch.constant.str "AssertionError: invalid shape dimensions"
    %str_1 = torch.constant.str "AssertionError: only one dimension can be inferred"
    %int-1 = torch.constant.int -1
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.int):
      %12 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.mul.int %arg3, %12 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%13 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    %2 = torch.prim.Uninitialized : !torch.int
    %3 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %4 = torch.derefine %none : !torch.none to !torch.optional<int>
    %5:2 = torch.prim.Loop %3, %true, init(%int1, %4) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.optional<int>):
      %12 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.eq.int %12, %int-1 : !torch.int, !torch.int -> !torch.bool
      %14:2 = torch.prim.If %13 -> (!torch.int, !torch.optional<int>) {
        %15 = torch.aten.__isnot__ %arg4, %none : !torch.optional<int>, !torch.none -> !torch.bool
        torch.prim.If %15 -> () {
          torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        %16 = torch.derefine %arg2 : !torch.int to !torch.optional<int>
        torch.prim.If.yield %arg3, %16 : !torch.int, !torch.optional<int>
      } else {
        %15 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %16 = torch.aten.ge.int %15, %int0 : !torch.int, !torch.int -> !torch.bool
        %17 = torch.prim.If %16 -> (!torch.int) {
          %18 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %19 = torch.aten.mul.int %arg3, %18 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %19 : !torch.int
        } else {
          torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
          torch.prim.If.yield %2 : !torch.int
        }
        torch.prim.If.yield %17, %arg4 : !torch.int, !torch.optional<int>
      }
      torch.prim.Loop.condition %true, iter(%14#0, %14#1 : !torch.int, !torch.optional<int>)
    } : (!torch.int, !torch.bool, !torch.int, !torch.optional<int>) -> (!torch.int, !torch.optional<int>)
    %6 = torch.aten.eq.int %1, %5#0 : !torch.int, !torch.int -> !torch.bool
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %12 = torch.aten.__isnot__ %5#1, %none : !torch.optional<int>, !torch.none -> !torch.bool
      %13 = torch.prim.If %12 -> (!torch.bool) {
        %15 = torch.prim.unchecked_cast %5#1 : !torch.optional<int> -> !torch.int
        %16 = torch.aten.gt.int %5#0, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %16 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %14 = torch.prim.If %13 -> (!torch.bool) {
        %15 = torch.prim.unchecked_cast %5#1 : !torch.optional<int> -> !torch.int
        %16 = torch.aten.remainder.int %1, %5#0 : !torch.int, !torch.int -> !torch.int
        %17 = torch.aten.eq.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %17 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %14 : !torch.bool
    }
    %8 = torch.aten.__not__ %7 : !torch.bool -> !torch.bool
    torch.prim.If %8 -> () {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    %9 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %10 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    torch.prim.Loop %10, %true, init() {
    ^bb0(%arg2: !torch.int):
      %12 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.append.t %9, %12 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %11 = torch.aten.__isnot__ %5#1, %none : !torch.optional<int>, !torch.none -> !torch.bool
    torch.prim.If %11 -> () {
      %12 = torch.prim.unchecked_cast %5#1 : !torch.optional<int> -> !torch.int
      %13 = torch.aten.floordiv.int %1, %5#0 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten._set_item.t %9, %12, %13 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    return %9 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.infer_size_impl(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: invalid shape"
    %false = torch.constant.bool false
    %str_0 = torch.constant.str "AssertionError: invalid shape dimensions"
    %str_1 = torch.constant.str "AssertionError: only one dimension can be inferred"
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
          torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
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
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    %7 = call @__torch__.torch.jit._shape_functions._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
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
  func.func @__torch__.torch.jit._shape_functions.expand(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
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
      %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %8 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      torch.prim.Loop %8, %true, init() {
      ^bb0(%arg2: !torch.int):
        %9 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.append.t %7, %9 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
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
  func.func @__torch__.torch.jit._shape_functions.expand_one_unused(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.any) -> !torch.list<int> {
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
      %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %8 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      torch.prim.Loop %8, %true, init() {
      ^bb0(%arg3: !torch.int):
        %9 = torch.aten.__getitem__.t %arg1, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.append.t %7, %9 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %7 : !torch.list<int>
    } else {
      %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %3, %true, init() {
      ^bb0(%arg3: !torch.int):
        %8 = torch.aten.sub.int %3, %int1 : !torch.int, !torch.int -> !torch.int
        %9 = torch.aten.sub.int %8, %arg3 : !torch.int, !torch.int -> !torch.int
        %10 = torch.aten.sub.int %4, %int1 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.sub.int %10, %9 : !torch.int, !torch.int -> !torch.int
        %12 = torch.aten.ge.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
        %13 = torch.prim.If %12 -> (!torch.int) {
          %20 = torch.aten.__getitem__.t %arg0, %11 : !torch.list<int>, !torch.int -> !torch.int
          torch.prim.If.yield %20 : !torch.int
        } else {
          torch.prim.If.yield %int1 : !torch.int
        }
        %14 = torch.aten.__getitem__.t %arg1, %arg3 : !torch.list<int>, !torch.int -> !torch.int
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
  func.func @__torch__.torch.jit._shape_functions.mean_dim(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.any) -> !torch.list<int> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg4: !torch.int):
      %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %3 = torch.prim.Loop %2, %true, init(%false) {
      ^bb0(%arg5: !torch.int, %arg6: !torch.bool):
        %4 = torch.aten.__getitem__.t %arg1, %arg5 : !torch.list<int>, !torch.int -> !torch.int
        %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %6 = torch.aten.le.int %5, %int0 : !torch.int, !torch.int -> !torch.bool
        %7 = torch.prim.If %6 -> (!torch.int) {
          torch.prim.If.yield %int1 : !torch.int
        } else {
          torch.prim.If.yield %5 : !torch.int
        }
        %8 = torch.aten.neg.int %7 : !torch.int -> !torch.int
        %9 = torch.aten.sub.int %7, %int1 : !torch.int, !torch.int -> !torch.int
        %10 = torch.aten.lt.int %4, %8 : !torch.int, !torch.int -> !torch.bool
        %11 = torch.prim.If %10 -> (!torch.bool) {
          torch.prim.If.yield %true : !torch.bool
        } else {
          %17 = torch.aten.gt.int %4, %9 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %17 : !torch.bool
        }
        %12 = torch.aten.__not__ %11 : !torch.bool -> !torch.bool
        torch.prim.If %12 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %13 = torch.aten.lt.int %4, %int0 : !torch.int, !torch.int -> !torch.bool
        %14 = torch.prim.If %13 -> (!torch.int) {
          %17 = torch.aten.add.int %4, %7 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %17 : !torch.int
        } else {
          torch.prim.If.yield %4 : !torch.int
        }
        %15 = torch.aten.eq.int %arg4, %14 : !torch.int, !torch.int -> !torch.bool
        %16 = torch.prim.If %15 -> (!torch.bool) {
          torch.prim.If.yield %true : !torch.bool
        } else {
          torch.prim.If.yield %arg6 : !torch.bool
        }
        torch.prim.Loop.condition %true, iter(%16 : !torch.bool)
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
  func.func @__torch__.torch.jit._shape_functions.max_dim(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %arg1 : (!torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %2, %true, init() {
    ^bb0(%arg3: !torch.int):
      %4 = torch.prim.Loop %int1, %true, init(%false) {
      ^bb0(%arg4: !torch.int, %arg5: !torch.bool):
        %5 = torch.aten.__getitem__.t %0, %arg4 : !torch.list<int>, !torch.int -> !torch.int
        %6 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
        %7 = torch.aten.le.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
        %8 = torch.prim.If %7 -> (!torch.int) {
          torch.prim.If.yield %int1 : !torch.int
        } else {
          torch.prim.If.yield %6 : !torch.int
        }
        %9 = torch.aten.neg.int %8 : !torch.int -> !torch.int
        %10 = torch.aten.sub.int %8, %int1 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.lt.int %5, %9 : !torch.int, !torch.int -> !torch.bool
        %12 = torch.prim.If %11 -> (!torch.bool) {
          torch.prim.If.yield %true : !torch.bool
        } else {
          %18 = torch.aten.gt.int %5, %10 : !torch.int, !torch.int -> !torch.bool
          torch.prim.If.yield %18 : !torch.bool
        }
        %13 = torch.aten.__not__ %12 : !torch.bool -> !torch.bool
        torch.prim.If %13 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %14 = torch.aten.lt.int %5, %int0 : !torch.int, !torch.int -> !torch.bool
        %15 = torch.prim.If %14 -> (!torch.int) {
          %18 = torch.aten.add.int %5, %8 : !torch.int, !torch.int -> !torch.int
          torch.prim.If.yield %18 : !torch.int
        } else {
          torch.prim.If.yield %5 : !torch.int
        }
        %16 = torch.aten.eq.int %arg3, %15 : !torch.int, !torch.int -> !torch.bool
        %17 = torch.prim.If %16 -> (!torch.bool) {
          torch.prim.If.yield %true : !torch.bool
        } else {
          torch.prim.If.yield %arg5 : !torch.bool
        }
        torch.prim.Loop.condition %true, iter(%17 : !torch.bool)
      } : (!torch.int, !torch.bool, !torch.bool) -> !torch.bool
      torch.prim.If %4 -> () {
        torch.prim.If %arg2 -> () {
          %5 = torch.aten.append.t %1, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        } else {
          torch.prim.If.yield
        }
        torch.prim.If.yield
      } else {
        %5 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %6 = torch.aten.append.t %1, %5 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %3 = torch.prim.TupleConstruct %1, %1 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %3 : !torch.tuple<list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.addmm(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.any, %arg4: !torch.any) -> !torch.list<int> {
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %str_0 = torch.constant.str "AssertionError: self must be a matrix"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: mat2 must be a matrix"
    %str_2 = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int2 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %6 = torch.aten.eq.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.prim.ListConstruct %7, %8 : (!torch.int, !torch.int) -> !torch.list<int>
    %10 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %11 = torch.prim.max.int %10, %int2 : !torch.int, !torch.int -> !torch.int
    %12 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %11, %true, init() {
    ^bb0(%arg5: !torch.int):
      %13 = torch.aten.sub.int %11, %int1 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.sub.int %13, %arg5 : !torch.int, !torch.int -> !torch.int
      %15 = torch.aten.sub.int %10, %int1 : !torch.int, !torch.int -> !torch.int
      %16 = torch.aten.sub.int %15, %14 : !torch.int, !torch.int -> !torch.int
      %17 = torch.aten.sub.int %int1, %14 : !torch.int, !torch.int -> !torch.int
      %18 = torch.aten.ge.int %16, %int0 : !torch.int, !torch.int -> !torch.bool
      %19 = torch.prim.If %18 -> (!torch.int) {
        %28 = torch.aten.__getitem__.t %arg0, %16 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %20 = torch.aten.ge.int %17, %int0 : !torch.int, !torch.int -> !torch.bool
      %21 = torch.prim.If %20 -> (!torch.int) {
        %28 = torch.aten.__getitem__.t %9, %17 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %28 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %22 = torch.aten.ne.int %19, %21 : !torch.int, !torch.int -> !torch.bool
      %23 = torch.prim.If %22 -> (!torch.bool) {
        %28 = torch.aten.ne.int %19, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %28 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %24 = torch.prim.If %23 -> (!torch.bool) {
        %28 = torch.aten.ne.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %28 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %24 -> () {
        %28 = torch.aten.format(%str, %19, %21, %arg5) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %29 = torch.aten.add.str %str_2, %28 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %29, %none : !torch.str, !torch.none
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      %25 = torch.aten.eq.int %19, %int1 : !torch.int, !torch.int -> !torch.bool
      %26 = torch.prim.If %25 -> (!torch.int) {
        torch.prim.If.yield %21 : !torch.int
      } else {
        torch.prim.If.yield %19 : !torch.int
      }
      %27 = torch.aten.append.t %12, %26 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %12 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.upsample_nearest2d(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<float>>) -> !torch.optional<list<int>> {
    %str = torch.constant.str "AssertionError: Either output_size or scale_factors must be presented"
    %str_0 = torch.constant.str "AssertionError: "
    %str_1 = torch.constant.str "AssertionError: Must specify exactly one of output_size and scale_factors"
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %2 = torch.aten.append.t %0, %1 : !torch.list<int>, !torch.int -> !torch.list<int>
    %3 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.append.t %0, %3 : !torch.list<int>, !torch.int -> !torch.list<int>
    %5 = torch.aten.__isnot__ %arg1, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.optional<list<int>>) {
      %7 = torch.prim.unchecked_cast %arg1 : !torch.optional<list<int>> -> !torch.list<int>
      %8 = torch.aten.__is__ %arg2, %none : !torch.optional<list<float>>, !torch.none -> !torch.bool
      torch.prim.If %8 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %9 = torch.aten.len.t %7 : !torch.list<int> -> !torch.int
      %10 = torch.aten.eq.int %9, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %10 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %11 = torch.aten.__getitem__.t %7, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %12 = torch.aten.append.t %0, %11 : !torch.list<int>, !torch.int -> !torch.list<int>
      %13 = torch.aten.__getitem__.t %7, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %14 = torch.aten.append.t %0, %13 : !torch.list<int>, !torch.int -> !torch.list<int>
      %15 = torch.derefine %0 : !torch.list<int> to !torch.optional<list<int>>
      torch.prim.If.yield %15 : !torch.optional<list<int>>
    } else {
      %7 = torch.aten.__isnot__ %arg2, %none : !torch.optional<list<float>>, !torch.none -> !torch.bool
      %8 = torch.prim.If %7 -> (!torch.optional<list<int>>) {
        %9 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<float>> -> !torch.list<float>
        %10 = torch.aten.__is__ %arg1, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
        torch.prim.If %10 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %11 = torch.aten.len.t %9 : !torch.list<float> -> !torch.int
        %12 = torch.aten.eq.int %11, %int2 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %12 -> () {
          torch.prim.If.yield
        } else {
          torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
          torch.prim.If.yield
        }
        %13 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
        %14 = torch.aten.__getitem__.t %9, %int0 : !torch.list<float>, !torch.int -> !torch.float
        %15 = torch.operator "aten.mul.int_float"(%13, %14) : (!torch.int, !torch.float) -> !torch.float
        %16 = torch.aten.Int.float %15 : !torch.float -> !torch.int
        %17 = torch.aten.append.t %0, %16 : !torch.list<int>, !torch.int -> !torch.list<int>
        %18 = torch.aten.__getitem__.t %arg0, %int3 : !torch.list<int>, !torch.int -> !torch.int
        %19 = torch.aten.__getitem__.t %9, %int1 : !torch.list<float>, !torch.int -> !torch.float
        %20 = torch.operator "aten.mul.int_float"(%18, %19) : (!torch.int, !torch.float) -> !torch.float
        %21 = torch.aten.Int.float %20 : !torch.float -> !torch.int
        %22 = torch.aten.append.t %0, %21 : !torch.list<int>, !torch.int -> !torch.list<int>
        %23 = torch.derefine %0 : !torch.list<int> to !torch.optional<list<int>>
        torch.prim.If.yield %23 : !torch.optional<list<int>>
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        %9 = torch.derefine %none : !torch.none to !torch.optional<list<int>>
        torch.prim.If.yield %9 : !torch.optional<list<int>>
      }
      torch.prim.If.yield %8 : !torch.optional<list<int>>
    }
    return %6 : !torch.optional<list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.argmax(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %str = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.__is__ %arg1, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.list<int>) {
      %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %2 : !torch.list<int>
    } else {
      %2 = torch.prim.unchecked_cast %arg1 : !torch.optional<int> -> !torch.int
      %3 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %4 = torch.aten.le.int %3, %int0 : !torch.int, !torch.int -> !torch.bool
      %5 = torch.prim.If %4 -> (!torch.int) {
        torch.prim.If.yield %int1 : !torch.int
      } else {
        torch.prim.If.yield %3 : !torch.int
      }
      %6 = torch.aten.neg.int %5 : !torch.int -> !torch.int
      %7 = torch.aten.sub.int %5, %int1 : !torch.int, !torch.int -> !torch.int
      %8 = torch.aten.lt.int %2, %6 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.bool) {
        torch.prim.If.yield %true : !torch.bool
      } else {
        %17 = torch.aten.gt.int %2, %7 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %17 : !torch.bool
      }
      %10 = torch.aten.__not__ %9 : !torch.bool -> !torch.bool
      torch.prim.If %10 -> () {
        torch.prim.If.yield
      } else {
        torch.prim.RaiseException %str, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %11 = torch.aten.lt.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
      %12 = torch.prim.If %11 -> (!torch.int) {
        %17 = torch.aten.add.int %2, %5 : !torch.int, !torch.int -> !torch.int
        torch.prim.If.yield %17 : !torch.int
      } else {
        torch.prim.If.yield %2 : !torch.int
      }
      %13 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %14 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %15 = torch.prim.ListConstruct %int9223372036854775807, %14 : (!torch.int, !torch.int) -> !torch.list<int>
      %16 = torch.prim.min.self_int %15 : !torch.list<int> -> !torch.int
      torch.prim.Loop %16, %true, init() {
      ^bb0(%arg3: !torch.int):
        %17 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %18 = torch.aten.eq.int %arg3, %12 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If %18 -> () {
          torch.prim.If %arg2 -> () {
            %19 = torch.aten.append.t %13, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
            torch.prim.If.yield
          } else {
            torch.prim.If.yield
          }
          torch.prim.If.yield
        } else {
          %19 = torch.aten.append.t %13, %17 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield
        }
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %13 : !torch.list<int>
    }
    return %1 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions._reduce_along_dim(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch.jit._shape_functions.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
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
  func.func @__torch__.torch.jit._shape_functions.bmm(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: mismatching contracting dimension"
    %str_0 = torch.constant.str "AssertionError: mismatching batch dimension"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: bmm only supports 3D tensors"
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
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
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %10 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %11 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %12 = torch.aten.__getitem__.t %arg1, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %13 = torch.prim.ListConstruct %10, %11, %12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    return %13 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions._shape_as_tensor(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.topk(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.tuple<list<int>, list<int>> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %str_0 = torch.constant.str "k ({}) is too big for dimension {} of size {}"
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.list<int>) {
      %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %4 : !torch.list<int>
    } else {
      %4 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.aten.le.int %arg1, %4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If %5 -> () {
        torch.prim.If.yield
      } else {
        %9 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.format(%str_0, %arg1, %arg2, %9) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %11 = torch.aten.add.str %str, %10 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %11, %none : !torch.str, !torch.none
        torch.prim.If.yield
      }
      %6 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %7 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      torch.prim.Loop %7, %true, init() {
      ^bb0(%arg3: !torch.int):
        %9 = torch.aten.__getitem__.t %arg0, %arg3 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.append.t %6, %9 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      %8 = torch.aten._set_item.t %6, %arg2, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield %6 : !torch.list<int>
    }
    %3 = torch.prim.TupleConstruct %2, %2 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %3 : !torch.tuple<list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.nll_loss_forward(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.int) -> !torch.tuple<list<int>, list<int>> {
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.lt.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.prim.If %2 -> (!torch.bool) {
      %16 = torch.aten.le.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %16 : !torch.bool
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
      %16 = torch.aten.eq.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %16 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %7 = torch.prim.If %6 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %16 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %17 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %18 = torch.aten.eq.int %16, %17 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %18 : !torch.bool
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
      %16 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      %17 = torch.aten.len.t %16 : !torch.list<int> -> !torch.int
      %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
      %19 = torch.prim.If %18 -> (!torch.bool) {
        %20 = torch.aten.__getitem__.t %16, %int0 : !torch.list<int>, !torch.int -> !torch.int
        %21 = torch.aten.eq.int %20, %8 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %21 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If.yield %19 : !torch.bool
    }
    torch.prim.If %11 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %12 = torch.aten.eq.int %arg3, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.bool) {
      %16 = torch.aten.eq.int %0, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %16 : !torch.bool
    } else {
      torch.prim.If.yield %false : !torch.bool
    }
    %14 = torch.prim.If %13 -> (!torch.list<int>) {
      %16 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
      %17 = torch.prim.ListConstruct %16 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %17 : !torch.list<int>
    } else {
      torch.prim.If.yield %9 : !torch.list<int>
    }
    %15 = torch.prim.TupleConstruct %14, %9 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %15 : !torch.tuple<list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.native_layer_norm(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
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
    ^bb0(%arg2: !torch.int):
      %10 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %11 = torch.aten.append.t %0, %10 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.__range_length %3, %5, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %6, %true, init() {
    ^bb0(%arg2: !torch.int):
      %10 = torch.aten.append.t %0, %int1 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %8 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %8, %true, init() {
    ^bb0(%arg2: !torch.int):
      %10 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %11 = torch.aten.append.t %7, %10 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %9 = torch.prim.TupleConstruct %7, %0, %0 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
    return %9 : !torch.tuple<list<int>, list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.native_batch_norm(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.If %arg5 -> (!torch.list<int>) {
      %4 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.prim.ListConstruct %4 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %4 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
      torch.prim.If.yield %4 : !torch.list<int>
    }
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %2, %true, init() {
    ^bb0(%arg6: !torch.int):
      %4 = torch.aten.__getitem__.t %arg0, %arg6 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.aten.append.t %1, %4 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %3 = torch.prim.TupleConstruct %1, %0, %0 : !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>>
    return %3 : !torch.tuple<list<int>, list<int>, list<int>>
  }
  func.func @__torch__.torch.jit._shape_functions.broadcast_three(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
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
    ^bb0(%arg3: !torch.int):
      %8 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
      %9 = torch.aten.sub.int %8, %arg3 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.sub.int %0, %int1 : !torch.int, !torch.int -> !torch.int
      %11 = torch.aten.sub.int %10, %9 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.sub.int %1, %int1 : !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.sub.int %12, %9 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.ge.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.int) {
        %24 = torch.aten.__getitem__.t %arg0, %11 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %24 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %16 = torch.aten.ge.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      %17 = torch.prim.If %16 -> (!torch.int) {
        %24 = torch.aten.__getitem__.t %arg1, %13 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %24 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %18 = torch.aten.ne.int %15, %17 : !torch.int, !torch.int -> !torch.bool
      %19 = torch.prim.If %18 -> (!torch.bool) {
        %24 = torch.aten.ne.int %15, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %24 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %20 = torch.prim.If %19 -> (!torch.bool) {
        %24 = torch.aten.ne.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %24 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %20 -> () {
        %24 = torch.aten.format(%str, %15, %17, %arg3) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %25 = torch.aten.add.str %str_0, %24 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %25, %none : !torch.str, !torch.none
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      %21 = torch.aten.eq.int %15, %int1 : !torch.int, !torch.int -> !torch.bool
      %22 = torch.prim.If %21 -> (!torch.int) {
        torch.prim.If.yield %17 : !torch.int
      } else {
        torch.prim.If.yield %15 : !torch.int
      }
      %23 = torch.aten.append.t %3, %22 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %4 = torch.aten.len.t %3 : !torch.list<int> -> !torch.int
    %5 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %6 = torch.prim.max.int %4, %5 : !torch.int, !torch.int -> !torch.int
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %6, %true, init() {
    ^bb0(%arg3: !torch.int):
      %8 = torch.aten.sub.int %6, %int1 : !torch.int, !torch.int -> !torch.int
      %9 = torch.aten.sub.int %8, %arg3 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.sub.int %4, %int1 : !torch.int, !torch.int -> !torch.int
      %11 = torch.aten.sub.int %10, %9 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.sub.int %5, %int1 : !torch.int, !torch.int -> !torch.int
      %13 = torch.aten.sub.int %12, %9 : !torch.int, !torch.int -> !torch.int
      %14 = torch.aten.ge.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.int) {
        %24 = torch.aten.__getitem__.t %3, %11 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %24 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %16 = torch.aten.ge.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
      %17 = torch.prim.If %16 -> (!torch.int) {
        %24 = torch.aten.__getitem__.t %arg2, %13 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %24 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %18 = torch.aten.ne.int %15, %17 : !torch.int, !torch.int -> !torch.bool
      %19 = torch.prim.If %18 -> (!torch.bool) {
        %24 = torch.aten.ne.int %15, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %24 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %20 = torch.prim.If %19 -> (!torch.bool) {
        %24 = torch.aten.ne.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %24 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %20 -> () {
        %24 = torch.aten.format(%str, %15, %17, %arg3) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %25 = torch.aten.add.str %str_0, %24 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %25, %none : !torch.str, !torch.none
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      %21 = torch.aten.eq.int %15, %int1 : !torch.int, !torch.int -> !torch.bool
      %22 = torch.prim.If %21 -> (!torch.int) {
        torch.prim.If.yield %17 : !torch.int
      } else {
        torch.prim.If.yield %15 : !torch.int
      }
      %23 = torch.aten.append.t %7, %22 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %7 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.broadcast_one_three(%arg0: !torch.list<int>, %arg1: !torch.any, %arg2: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_0 = torch.constant.str "AssertionError: "
    %none = torch.constant.none
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %2 = torch.prim.max.int %0, %1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    torch.prim.Loop %2, %true, init() {
    ^bb0(%arg3: !torch.int):
      %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
      %5 = torch.aten.sub.int %4, %arg3 : !torch.int, !torch.int -> !torch.int
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
        %20 = torch.aten.__getitem__.t %arg2, %9 : !torch.list<int>, !torch.int -> !torch.int
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
        %20 = torch.aten.format(%str, %11, %13, %arg3) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
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
  func.func @__torch__.torch.jit._shape_functions.broadcast_inplace(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: "
    %str_1 = torch.constant.str "The dims of tensor b ({}) must be less than or equal tothe dims of tensor a ({}) "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %2 = torch.aten.gt.int %1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      %5 = torch.aten.format(%str_1, %1, %0) : !torch.str, !torch.int, !torch.int -> !torch.str
      %6 = torch.aten.add.str %str_0, %5 : !torch.str, !torch.str -> !torch.str
      torch.prim.RaiseException %6, %none : !torch.str, !torch.none
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    torch.prim.Loop %0, %true, init() {
    ^bb0(%arg2: !torch.int):
      %5 = torch.aten.sub.int %1, %0 : !torch.int, !torch.int -> !torch.int
      %6 = torch.aten.add.int %5, %arg2 : !torch.int, !torch.int -> !torch.int
      %7 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %8 = torch.aten.ge.int %6, %int0 : !torch.int, !torch.int -> !torch.bool
      %9 = torch.prim.If %8 -> (!torch.int) {
        %12 = torch.aten.__getitem__.t %arg1, %6 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %12 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %10 = torch.aten.ne.int %7, %9 : !torch.int, !torch.int -> !torch.bool
      %11 = torch.prim.If %10 -> (!torch.bool) {
        %12 = torch.aten.ne.int %9, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %12 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %11 -> () {
        %12 = torch.aten.format(%str, %7, %9, %arg2) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %13 = torch.aten.add.str %str_0, %12 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %13, %none : !torch.str, !torch.none
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %4, %true, init() {
    ^bb0(%arg2: !torch.int):
      %5 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %6 = torch.aten.append.t %3, %5 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %3 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.nonzero_lower_bound(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.ListConstruct %int0, %0 : (!torch.int, !torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @__torch__.torch.jit._shape_functions.nonzero_upper_bound(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.Loop %0, %true, init(%int1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.int):
      %4 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.aten.mul.int %arg2, %4 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop.condition %true, iter(%5 : !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> !torch.int
    %2 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %3 = torch.prim.ListConstruct %1, %2 : (!torch.int, !torch.int) -> !torch.list<int>
    return %3 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.triu"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.tanh"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.erf"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sigmoid"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.hardsigmoid"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.softplus"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.square"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.hardswish"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.silu"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.exp"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.expm1"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sin"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.cos"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.hardtanh"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sqrt"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.neg"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.floor"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.detach"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.log2"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.log1p"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.rsqrt"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.abs"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.reciprocal"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.tanh_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.gelu_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.str) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.ceil"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.log"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.relu"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._softmax"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.softmax.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._log_softmax"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.log_softmax.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.clamp"(%arg0: !torch.list<int>, %arg1: !torch.optional<float>, %arg2: !torch.optional<float>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.clamp_min"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.clamp_max"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.rsub.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.remainder.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.to.dtype"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool, %arg3: !torch.bool, %arg4: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.to.dtype_layout"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.bool, %arg6: !torch.bool, %arg7: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.to.other"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.bool, %arg4: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.type_as"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.dropout"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.gelu"(%arg0: !torch.list<int>, %arg1: !torch.str) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.contiguous"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.clone"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._log_softmax_backward_data"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.eq.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.ne.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.gt.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.ge.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.le.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.lt.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.add.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sub.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.mul.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.div.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.floor_divide.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.pow.Tensor_Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.leaky_relu"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.gather"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg2) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.layer_norm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.float, %arg5: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._softmax_backward_data"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg1) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.any"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.all"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.max"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sum"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.mean"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.var"(%arg0: !torch.list<int>, %arg1: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.var.dim"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.bool) -> !torch.list<int> {
    %none = torch.constant.none
    %0 = torch.derefine %none : !torch.none to !torch.any
    %1 = call @__torch__.torch.jit._shape_functions.mean_dim(%arg0, %arg1, %arg3, %0) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.std"(%arg0: !torch.list<int>, %arg1: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.argmax"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %none = torch.constant.none
    %0 = torch.aten.__is__ %arg1, %none : !torch.optional<int>, !torch.none -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.list<int>) {
      %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.If.yield %2 : !torch.list<int>
    } else {
      %2 = torch.prim.unchecked_cast %arg1 : !torch.optional<int> -> !torch.int
      %3 = func.call @__torch__._reduce_along_dim(%arg0, %2, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
      torch.prim.If.yield %3 : !torch.list<int>
    }
    return %1 : !torch.list<int>
  }
  func.func @__torch__._reduce_along_dim(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = call @__torch__.torch.jit._shape_functions.maybe_wrap_dim(%arg1, %0, %true) : (!torch.int, !torch.int, !torch.bool) -> !torch.int
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
  func.func @"__torch_mlir_shape_fn.aten.any.dim"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__._reduce_along_dim(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.max.dim"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %0 = call @__torch__._reduce_along_dim(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.bool) -> !torch.list<int>
    %1 = torch.prim.TupleConstruct %0, %0 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %1 : !torch.tuple<list<int>, list<int>>
  }
  func.func @"__torch_mlir_shape_fn.aten.mean.dim"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool, %arg3: !torch.optional<int>) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %1 = call @__torch__.torch.jit._shape_functions.mean_dim(%arg0, %arg1, %arg2, %0) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sum.dim_IntList"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.bool, %arg3: !torch.optional<int>) -> !torch.list<int> {
    %none = torch.constant.none
    %0 = torch.aten.__is__ %arg1, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.list<int>) {
      %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %3 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
      %4 = func.call @__torch__.torch.jit._shape_functions.mean_dim(%arg0, %2, %arg2, %3) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
      torch.prim.If.yield %4 : !torch.list<int>
    } else {
      %2 = torch.prim.unchecked_cast %arg1 : !torch.optional<list<int>> -> !torch.list<int>
      %3 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
      %4 = func.call @__torch__.torch.jit._shape_functions.mean_dim(%arg0, %2, %arg2, %3) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
      torch.prim.If.yield %4 : !torch.list<int>
    }
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.permute"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.permute(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.transpose.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.transpose(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.t"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = call @__torch__.torch.jit._shape_functions.transpose(%arg0, %int0, %int1) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.numpy_T"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    torch.prim.Loop %1, %true, init() {
    ^bb0(%arg1: !torch.int):
      %2 = torch.aten.__getitem__.t %arg0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
      torch.aten.insert.t %0, %int0, %2 : !torch.list<int>, !torch.int, !torch.int
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.matmul"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.matmul(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.mm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.mm(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.addmm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float, %arg4: !torch.float) -> !torch.list<int> {
    %0 = torch.derefine %arg3 : !torch.float to !torch.any
    %1 = torch.derefine %arg4 : !torch.float to !torch.any
    %2 = call @__torch__.torch.jit._shape_functions.addmm(%arg0, %arg1, %arg2, %0, %1) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.any, !torch.any) -> !torch.list<int>
    return %2 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bmm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: mismatching contracting dimension"
    %str_0 = torch.constant.str "AssertionError: mismatching batch dimension"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: bmm only supports 3D tensors"
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
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
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %10 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %11 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %12 = torch.aten.__getitem__.t %arg1, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %13 = torch.prim.ListConstruct %10, %11, %12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    return %13 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.baddbmm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float, %arg4: !torch.float) -> !torch.list<int> {
    %str = torch.constant.str "AssertionError: mismatching contracting dimension"
    %str_0 = torch.constant.str "AssertionError: mismatching batch dimension"
    %none = torch.constant.none
    %str_1 = torch.constant.str "AssertionError: baddbmm only supports 3D tensors"
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %3 = torch.aten.eq.int %2, %int3 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %3 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %4 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %5 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %6 = torch.aten.eq.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.__getitem__.t %arg1, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %8 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %9 = torch.aten.eq.int %7, %8 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %9 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %10 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %11 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %12 = torch.aten.__getitem__.t %arg2, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %13 = torch.prim.ListConstruct %10, %11, %12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    return %13 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.embedding"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.embedding(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.list<int>, !torch.list<int>, !torch.int, !torch.bool, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.repeat"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
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
      %7 = func.call @__torch__.torch.jit._shape_functions._copy(%arg0) : (!torch.list<int>) -> !torch.list<int>
      torch.prim.If.yield %7 : !torch.list<int>
    } else {
      %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
      %8 = torch.aten.sub.int %3, %4 : !torch.int, !torch.int -> !torch.int
      torch.prim.Loop %8, %true, init() {
      ^bb0(%arg2: !torch.int):
        %9 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.append.t %7, %9 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.Loop %4, %true, init() {
      ^bb0(%arg2: !torch.int):
        %9 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %10 = torch.aten.add.int %arg2, %8 : !torch.int, !torch.int -> !torch.int
        %11 = torch.aten.__getitem__.t %arg1, %10 : !torch.list<int>, !torch.int -> !torch.int
        %12 = torch.aten.mul.int %9, %11 : !torch.int, !torch.int -> !torch.int
        %13 = torch.aten.append.t %7, %12 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %7 : !torch.list<int>
    }
    return %6 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.expand"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.expand(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.expand_as"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg1) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.broadcast_to"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.expand(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.view"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.view(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.reshape"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.view(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._reshape_alias"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.view(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._unsafe_view"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.resize_"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.max_pool2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.max_pool2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.max_pool2d_with_indices"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %0 = call @__torch__.torch.jit._shape_functions.max_pool2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.list<int>
    %1 = torch.prim.TupleConstruct %0, %0 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %1 : !torch.tuple<list<int>, list<int>>
  }
  func.func @"__torch_mlir_shape_fn.aten.max_pool2d_with_indices_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.bool, %arg7: !torch.list<int>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.avg_pool2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.bool, %arg5: !torch.bool, %arg6: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.avg_pool2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.optional<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @__torch__.avg_pool2d(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.bool, %arg5: !torch.bool, %arg6: !torch.optional<int>) -> !torch.list<int> {
    %int-1 = torch.constant.int -1
    %int-2 = torch.constant.int -2
    %int-3 = torch.constant.int -3
    %int-4 = torch.constant.int -4
    %str = torch.constant.str "AssertionError: "
    %str_0 = torch.constant.str "AssertionError: avg_pool2d: padding must be either be a single int, or a tuple of two ints"
    %str_1 = torch.constant.str "AssertionError: avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    %none = torch.constant.none
    %str_2 = torch.constant.str "AssertionError: avg_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.eq.int %0, %int1 : !torch.int, !torch.int -> !torch.bool
    %2 = torch.prim.If %1 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %39 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %40 : !torch.bool
    }
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_2, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %5 = torch.aten.eq.int %4, %int1 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.prim.If %5 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %39 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %39 : !torch.int
    }
    %7 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %39 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int1 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %40 : !torch.bool
    }
    %10 = torch.prim.If %9 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %39 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %40 : !torch.bool
    }
    torch.prim.If %10 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_1, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %11 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %12 = torch.aten.eq.int %11, %int0 : !torch.int, !torch.int -> !torch.bool
    %13 = torch.prim.If %12 -> (!torch.int) {
      torch.prim.If.yield %3 : !torch.int
    } else {
      %39 = torch.aten.__getitem__.t %arg2, %int0 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %39 : !torch.int
    }
    %14 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
    %15 = torch.aten.eq.int %14, %int0 : !torch.int, !torch.int -> !torch.bool
    %16 = torch.prim.If %15 -> (!torch.int) {
      torch.prim.If.yield %6 : !torch.int
    } else {
      %39 = torch.aten.len.t %arg2 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int1 : !torch.int, !torch.int -> !torch.bool
      %41 = torch.prim.If %40 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        %42 = torch.aten.__getitem__.t %arg2, %int1 : !torch.list<int>, !torch.int -> !torch.int
        torch.prim.If.yield %42 : !torch.int
      }
      torch.prim.If.yield %41 : !torch.int
    }
    %17 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %18 = torch.aten.eq.int %17, %int1 : !torch.int, !torch.int -> !torch.bool
    %19 = torch.prim.If %18 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %39 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int2 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %40 : !torch.bool
    }
    torch.prim.If %19 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %20 = torch.aten.__getitem__.t %arg3, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %21 = torch.aten.len.t %arg3 : !torch.list<int> -> !torch.int
    %22 = torch.aten.eq.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
    %23 = torch.prim.If %22 -> (!torch.int) {
      torch.prim.If.yield %20 : !torch.int
    } else {
      %39 = torch.aten.__getitem__.t %arg3, %int1 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %39 : !torch.int
    }
    %24 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %25 = torch.aten.eq.int %24, %int3 : !torch.int, !torch.int -> !torch.bool
    %26 = torch.prim.If %25 -> (!torch.bool) {
      torch.prim.If.yield %true : !torch.bool
    } else {
      %39 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %40 = torch.aten.eq.int %39, %int4 : !torch.int, !torch.int -> !torch.bool
      torch.prim.If.yield %40 : !torch.bool
    }
    torch.prim.If %26 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %27 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %28 = torch.aten.eq.int %27, %int4 : !torch.int, !torch.int -> !torch.bool
    %29 = torch.prim.If %28 -> (!torch.int) {
      %39 = torch.aten.__getitem__.t %arg0, %int-4 : !torch.list<int>, !torch.int -> !torch.int
      torch.prim.If.yield %39 : !torch.int
    } else {
      torch.prim.If.yield %int1 : !torch.int
    }
    %30 = torch.aten.__getitem__.t %arg0, %int-3 : !torch.list<int>, !torch.int -> !torch.int
    %31 = torch.aten.__getitem__.t %arg0, %int-2 : !torch.list<int>, !torch.int -> !torch.int
    %32 = torch.aten.__getitem__.t %arg0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
    %33 = call @__torch__.torch.jit._shape_functions.pooling_output_shape(%31, %3, %20, %13, %int1, %arg4) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %34 = call @__torch__.torch.jit._shape_functions.pooling_output_shape(%32, %6, %23, %16, %int1, %arg4) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool) -> !torch.int
    %35 = call @__torch__.torch.jit._shape_functions.pool2d_shape_check(%arg0, %3, %6, %13, %16, %20, %23, %int1, %int1, %30, %31, %32, %33, %34) : (!torch.list<int>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.none
    %36 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %37 = torch.aten.eq.int %36, %int3 : !torch.int, !torch.int -> !torch.bool
    %38 = torch.prim.If %37 -> (!torch.list<int>) {
      %39 = torch.prim.ListConstruct %30, %33, %34 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %39 : !torch.list<int>
    } else {
      %39 = torch.prim.ListConstruct %29, %30, %33, %34 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
      torch.prim.If.yield %39 : !torch.list<int>
    }
    return %38 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.adaptive_avg_pool2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.adaptive_avg_pool2d(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.flatten.using_ints"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.flatten(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.linear"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.linear(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.zeros"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.ones"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.empty.memory_format"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.full"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.full_like"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>, %arg6: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.zeros_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.ones_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.empty_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.new_zeros"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.new_ones"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.new_empty"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    return %arg1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._to_copy"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.bool, %arg6: !torch.optional<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.masked_fill.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.zero"(%arg0: !torch.list<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.fill.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.copy"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.uniform"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float, %arg3: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bernoulli.float"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bernoulli.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.index_put_impl"(%arg0: !torch.list<int>, %arg1: !torch.list<optional<list<int>>>, %arg2: !torch.list<int>, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bernoulli"(%arg0: !torch.list<int>, %arg1: !torch.any) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.rand_like"(%arg0: !torch.list<int>, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>, %arg5: !torch.optional<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.arange.start_step"(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.float, %arg3: !torch.optional<int>, %arg4: !torch.optional<int>, %arg5: !torch.optional<Device>, %arg6: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg0 : !torch.float to !torch.union<float, int>
    %1 = torch.derefine %arg1 : !torch.float to !torch.union<float, int>
    %2 = torch.derefine %arg2 : !torch.float to !torch.union<float, int>
    %3 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %4 = torch.derefine %arg4 : !torch.optional<int> to !torch.any
    %5 = torch.derefine %arg5 : !torch.optional<Device> to !torch.any
    %6 = torch.derefine %arg6 : !torch.optional<bool> to !torch.any
    %7 = call @__torch__.torch.jit._shape_functions.arange_start_step(%0, %1, %2, %3, %4, %5, %6) : (!torch.union<float, int>, !torch.union<float, int>, !torch.union<float, int>, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %7 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.arange.start"(%arg0: !torch.float, %arg1: !torch.float, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.optional<Device>, %arg5: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg0 : !torch.float to !torch.union<float, int>
    %1 = torch.derefine %arg1 : !torch.float to !torch.union<float, int>
    %2 = torch.derefine %arg2 : !torch.optional<int> to !torch.any
    %3 = torch.derefine %arg3 : !torch.optional<int> to !torch.any
    %4 = torch.derefine %arg4 : !torch.optional<Device> to !torch.any
    %5 = torch.derefine %arg5 : !torch.optional<bool> to !torch.any
    %6 = call @__torch__.torch.jit._shape_functions.arange_start(%0, %1, %2, %3, %4, %5) : (!torch.union<float, int>, !torch.union<float, int>, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %6 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.arange"(%arg0: !torch.float, %arg1: !torch.optional<int>, %arg2: !torch.optional<int>, %arg3: !torch.optional<Device>, %arg4: !torch.optional<bool>) -> !torch.list<int> {
    %0 = torch.derefine %arg0 : !torch.float to !torch.union<float, int>
    %1 = torch.derefine %arg1 : !torch.optional<int> to !torch.any
    %2 = torch.derefine %arg2 : !torch.optional<int> to !torch.any
    %3 = torch.derefine %arg3 : !torch.optional<Device> to !torch.any
    %4 = torch.derefine %arg4 : !torch.optional<bool> to !torch.any
    %5 = call @__torch__.torch.jit._shape_functions.arange_end(%0, %1, %2, %3, %4) : (!torch.union<float, int>, !torch.any, !torch.any, !torch.any, !torch.any) -> !torch.list<int>
    return %5 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.add.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.sub.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.mul.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.div.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.div.Tensor_mode"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<str>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.floor_divide"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.__and__.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.minimum"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.maximum"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bitwise_and.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.logical_or"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.threshold"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.threshold_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.eq.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.gt.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.lt.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.unsqueeze"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unsqueeze(%arg0, %arg1) : (!torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.squeeze"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.squeeze_nodim(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.squeeze.dim"(%arg0: !torch.list<int>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.squeeze(%arg0, %arg1) : (!torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.prim.NumToTensor.Scalar"(%arg0: !torch.float) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.tensor.float"(%arg0: !torch.float, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.tensor.int"(%arg0: !torch.int, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.tensor.bool"(%arg0: !torch.bool, %arg1: !torch.optional<int>, %arg2: !torch.optional<Device>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._shape_as_tensor"(%arg0: !torch.list<int>) -> !torch.list<int> {
    %0 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.where.self"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.where.Scalar"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.where.ScalarOther"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.where.ScalarSelf"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.lerp.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.addcmul"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.addcdiv"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.broadcast(%arg1, %arg2) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    %1 = call @__torch__.torch.jit._shape_functions.broadcast(%arg0, %0) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.topk"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %str_0 = torch.constant.str "k ({}) is too big for dimension {} of size {}"
    %0 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
    %1 = torch.aten.le.int %arg1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      %4 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
      %5 = torch.aten.format(%str_0, %arg1, %arg2, %4) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
      %6 = torch.aten.add.str %str, %5 : !torch.str, !torch.str -> !torch.str
      torch.prim.RaiseException %6, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %2 = torch.aten._set_item.t %arg0, %arg2, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    %3 = torch.prim.TupleConstruct %arg0, %arg0 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %3 : !torch.tuple<list<int>, list<int>>
  }
  func.func @"__torch_mlir_shape_fn.aten.conv2d"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.conv2d(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.convolution"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.bool, %arg7: !torch.list<int>, %arg8: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.conv_output_size(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg8) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten._convolution"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.bool, %arg7: !torch.list<int>, %arg8: !torch.int, %arg9: !torch.bool, %arg10: !torch.bool, %arg11: !torch.bool, %arg12: !torch.bool) -> !torch.list<int> {
    %0 = call @"__torch_mlir_shape_fn.aten.convolution"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (!torch.list<int>, !torch.list<int>, !torch.optional<list<int>>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.flip"(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.batch_norm"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.slice.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.optional<int>, %arg3: !torch.optional<int>, %arg4: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.slice(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.list<int>, !torch.int, !torch.optional<int>, !torch.optional<int>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.slice_scatter"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.optional<int>, %arg4: !torch.optional<int>, %arg5: !torch.int) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.select.int"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.select(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.select_scatter"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.list<int> {
    return %arg0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.index_select"(%arg0: !torch.list<int>, %arg1: !torch.int, %arg2: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.index_select(%arg0, %arg1, %arg2) : (!torch.list<int>, !torch.int, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.index_put"(%arg0: !torch.list<int>, %arg1: !torch.list<optional<list<int>>>, %arg2: !torch.list<int>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.index_put.hacked_twin"(%arg0: !torch.list<int>, %arg1: !torch.list<list<int>>, %arg2: !torch.list<int>, %arg3: !torch.bool) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg0) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.nll_loss_forward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.int, %arg4: !torch.int) -> !torch.tuple<list<int>, list<int>> {
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
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
  func.func @"__torch_mlir_shape_fn.aten.nll_loss_backward"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.optional<list<int>>, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.list<int>) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.unary(%arg1) : (!torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.native_layer_norm"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.float) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: "
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
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
  func.func @"__torch_mlir_shape_fn.aten.native_batch_norm"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.optional<list<int>>, %arg3: !torch.optional<list<int>>, %arg4: !torch.optional<list<int>>, %arg5: !torch.bool, %arg6: !torch.float, %arg7: !torch.float) -> !torch.tuple<list<int>, list<int>, list<int>> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
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
  func.func @"__torch_mlir_shape_fn.aten.constant_pad_nd"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.float) -> !torch.list<int> {
    %0 = call @__torch__.pad_shape_fn(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @__torch__.pad_shape_fn(%arg0: !torch.list<int>, %arg1: !torch.list<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %str = torch.constant.str "AssertionError: Number of padded dimensions must be less than or equal to the input dimension"
    %none = torch.constant.none
    %str_0 = torch.constant.str "AssertionError: Must have paired low-high pad amount values"
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %1 = torch.aten.remainder.int %0, %int2 : !torch.int, !torch.int -> !torch.int
    %2 = torch.aten.eq.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %2 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str_0, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %3 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %4 = torch.aten.floordiv.int %3, %int2 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.aten.le.int %4, %5 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %6 -> () {
      torch.prim.If.yield
    } else {
      torch.prim.RaiseException %str, %none : !torch.str, !torch.none
      torch.prim.If.yield
    }
    %7 = torch.aten.len.t %arg1 : !torch.list<int> -> !torch.int
    %8 = torch.aten.floordiv.int %7, %int2 : !torch.int, !torch.int -> !torch.int
    torch.prim.Loop %8, %true, init() {
    ^bb0(%arg2: !torch.int):
      %9 = torch.aten.add.int %arg2, %int1 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.neg.int %9 : !torch.int -> !torch.int
      %11 = torch.aten.mul.int %int2, %arg2 : !torch.int, !torch.int -> !torch.int
      %12 = torch.aten.__getitem__.t %arg1, %11 : !torch.list<int>, !torch.int -> !torch.int
      %13 = torch.aten.mul.int %int2, %arg2 : !torch.int, !torch.int -> !torch.int
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
  func.func @"__torch_mlir_shape_fn.aten.pad"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.str, %arg3: !torch.optional<float>) -> !torch.list<int> {
    %0 = call @__torch__.pad_shape_fn(%arg0, %arg1) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.index.Tensor"(%arg0: !torch.list<int>, %arg1: !torch.list<optional<list<int>>>) -> !torch.list<int> {
    %false = torch.constant.bool false
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %str = torch.constant.str "AssertionError: More indices than dimensions to index"
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int9223372036854775807 = torch.constant.int 9223372036854775807
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
    %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %5 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
    %6 = torch.prim.Loop %5, %true, init(%3) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.list<int>):
      %10 = torch.aten.len.t %arg1 : !torch.list<optional<list<int>>> -> !torch.int
      %11 = torch.aten.ge.int %arg2, %10 : !torch.int, !torch.int -> !torch.bool
      %12 = torch.prim.If %11 -> (!torch.list<int>) {
        %13 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
        %14 = torch.aten.append.t %4, %13 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.If.yield %arg3 : !torch.list<int>
      } else {
        %13 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<optional<list<int>>>, !torch.int -> !torch.optional<list<int>>
        %14 = torch.aten.__isnot__ %13, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
        %15 = torch.prim.If %14 -> (!torch.list<int>) {
          %16 = torch.prim.unchecked_cast %13 : !torch.optional<list<int>> -> !torch.list<int>
          %17 = func.call @__torch__.torch.jit._shape_functions.broadcast(%arg3, %16) : (!torch.list<int>, !torch.list<int>) -> !torch.list<int>
          torch.prim.If.yield %17 : !torch.list<int>
        } else {
          %16 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %17 = torch.aten.append.t %4, %16 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.If.yield %arg3 : !torch.list<int>
        }
        torch.prim.If.yield %15 : !torch.list<int>
      }
      torch.prim.Loop.condition %true, iter(%12 : !torch.list<int>)
    } : (!torch.int, !torch.bool, !torch.list<int>) -> !torch.list<int>
    %7 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
    %8 = torch.aten.eq.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
    %9 = torch.prim.If %8 -> (!torch.list<int>) {
      torch.prim.If.yield %6 : !torch.list<int>
    } else {
      %10 = torch.aten.len.t %arg1 : !torch.list<optional<list<int>>> -> !torch.int
      %11 = torch.prim.ListConstruct %int9223372036854775807, %10 : (!torch.int, !torch.int) -> !torch.list<int>
      %12 = torch.prim.min.self_int %11 : !torch.list<int> -> !torch.int
      %13:3 = torch.prim.Loop %12, %true, init(%true, %int-1, %int-1) {
      ^bb0(%arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.int, %arg5: !torch.int):
        %16 = torch.aten.__getitem__.t %arg1, %arg2 : !torch.list<optional<list<int>>>, !torch.int -> !torch.optional<list<int>>
        %17 = torch.aten.__isnot__ %16, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
        %18:3 = torch.prim.If %17 -> (!torch.bool, !torch.int, !torch.int) {
          %19 = torch.aten.eq.int %arg4, %int-1 : !torch.int, !torch.int -> !torch.bool
          %20:3 = torch.prim.If %19 -> (!torch.bool, !torch.int, !torch.int) {
            torch.prim.If.yield %arg3, %arg2, %arg2 : !torch.bool, !torch.int, !torch.int
          } else {
            %21 = torch.aten.sub.int %arg2, %arg5 : !torch.int, !torch.int -> !torch.int
            %22 = torch.aten.ne.int %21, %int1 : !torch.int, !torch.int -> !torch.bool
            %23 = torch.prim.If %22 -> (!torch.bool) {
              torch.prim.If.yield %false : !torch.bool
            } else {
              torch.prim.If.yield %arg3 : !torch.bool
            }
            torch.prim.If.yield %23, %arg4, %arg5 : !torch.bool, !torch.int, !torch.int
          }
          torch.prim.If.yield %20#0, %20#1, %20#2 : !torch.bool, !torch.int, !torch.int
        } else {
          torch.prim.If.yield %arg3, %arg4, %arg5 : !torch.bool, !torch.int, !torch.int
        }
        torch.prim.Loop.condition %true, iter(%18#0, %18#1, %18#2 : !torch.bool, !torch.int, !torch.int)
      } : (!torch.int, !torch.bool, !torch.bool, !torch.int, !torch.int) -> (!torch.bool, !torch.int, !torch.int)
      %14 = torch.aten.__not__ %13#0 : !torch.bool -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.list<int>) {
        %16 = torch.aten.add.t %6, %4 : !torch.list<int>, !torch.list<int> -> !torch.list<int>
        torch.prim.If.yield %16 : !torch.list<int>
      } else {
        %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
        torch.prim.Loop %13#1, %true, init() {
        ^bb0(%arg2: !torch.int):
          %20 = torch.aten.__getitem__.t %4, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %21 = torch.aten.append.t %16, %20 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        %17 = torch.aten.len.t %6 : !torch.list<int> -> !torch.int
        torch.prim.Loop %17, %true, init() {
        ^bb0(%arg2: !torch.int):
          %20 = torch.aten.__getitem__.t %6, %arg2 : !torch.list<int>, !torch.int -> !torch.int
          %21 = torch.aten.append.t %16, %20 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        %18 = torch.aten.len.t %4 : !torch.list<int> -> !torch.int
        %19 = torch.aten.__range_length %13#1, %18, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
        torch.prim.Loop %19, %true, init() {
        ^bb0(%arg2: !torch.int):
          %20 = torch.aten.__derive_index %arg2, %13#1, %int1 : !torch.int, !torch.int, !torch.int -> !torch.int
          %21 = torch.aten.__getitem__.t %4, %20 : !torch.list<int>, !torch.int -> !torch.int
          %22 = torch.aten.append.t %16, %21 : !torch.list<int>, !torch.int -> !torch.list<int>
          torch.prim.Loop.condition %true, iter()
        } : (!torch.int, !torch.bool) -> ()
        torch.prim.If.yield %16 : !torch.list<int>
      }
      torch.prim.If.yield %15 : !torch.list<int>
    }
    return %9 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.cat"(%arg0: !torch.list<list<int>>, %arg1: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.torch.jit._shape_functions.cat(%arg0, %arg1) : (!torch.list<list<int>>, !torch.int) -> !torch.list<int>
    return %0 : !torch.list<int>
  }
  func.func @"__torch_mlir_shape_fn.aten.bincount"(%arg0: !torch.list<int>, %arg1: !torch.optional<list<int>>, %arg2: !torch.int) -> !torch.list<int> {
    %0 = call @__torch__.hacky_get_unknown_dimension_size() : () -> !torch.int
    %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    return %1 : !torch.list<int>
  }
  func.func @__torch__.hacky_get_unknown_dimension_size() -> !torch.int {
    %0 = torch.prim.CreateObject !torch.nn.Module<"__torch__.DummyClassType">
    %1 = torch.prim.CallMethod %0["__init__"] () : !torch.nn.Module<"__torch__.DummyClassType">, () -> !torch.none
    %2 = torch.operator "prim.id"(%0) : (!torch.nn.Module<"__torch__.DummyClassType">) -> !torch.int
    return %2 : !torch.int
  }
  func.func @__torch__.DummyClassType.__init__(%arg0: !torch.nn.Module<"__torch__.DummyClassType">) -> !torch.none {
    %none = torch.constant.none
    return %none : !torch.none
  }
  func.func @"__torch_mlir_shape_fn.aten.linalg_vector_norm"(%arg0: !torch.list<int>, %arg1: !torch.float, %arg2: !torch.optional<list<int>>, %arg3: !torch.bool, %arg4: !torch.optional<int>) -> !torch.list<int> {
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.__is__ %arg2, %none : !torch.optional<list<int>>, !torch.none -> !torch.bool
    %1 = torch.prim.If %0 -> (!torch.list<int>) {
      %4 = torch.aten.len.t %arg0 : !torch.list<int> -> !torch.int
      %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
      torch.prim.Loop %4, %true, init() {
      ^bb0(%arg5: !torch.int):
        %6 = torch.aten.append.t %5, %arg5 : !torch.list<int>, !torch.int -> !torch.list<int>
        torch.prim.Loop.condition %true, iter()
      } : (!torch.int, !torch.bool) -> ()
      torch.prim.If.yield %5 : !torch.list<int>
    } else {
      %4 = torch.prim.unchecked_cast %arg2 : !torch.optional<list<int>> -> !torch.list<int>
      torch.prim.If.yield %4 : !torch.list<int>
    }
    %2 = torch.derefine %arg4 : !torch.optional<int> to !torch.any
    %3 = call @__torch__.torch.jit._shape_functions.mean_dim(%arg0, %1, %arg3, %2) : (!torch.list<int>, !torch.list<int>, !torch.bool, !torch.any) -> !torch.list<int>
    return %3 : !torch.list<int>
  }
}
)mlir");
#pragma clang diagnostic pop
  return shapeLib;
}
