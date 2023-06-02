//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_DIALECT_TCP_IR_TCPOPS_H_
#define TORCH_MLIR_DIALECTS_DIALECT_TCP_IR_TCPOPS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpAttrs.h.inc"
#define GET_OP_CLASSES
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h.inc"

#endif // TORCH_MLIR_DIALECTS_DIALECT_TCP_IR_TCPOPS_H_
