//===- node_importer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H

#include <memory>

#include "pybind.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

using CreateTerminatorFn =
    std::function<void(c10::ArrayRef<MlirValue>, MlirBlock)>;

MlirBlock importBlock(MlirContext context, torch::jit::Block *jitBlock,
                      CreateTerminatorFn createTerminator);

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H
