//===- LivenessReport.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_LIVENESSREPORT_H
#define NPCOMP_DIALECT_ATEN_LIVENESSREPORT_H

#include <string>

namespace mlir {
namespace NPCOMP {
namespace aten {

struct LivenessReport {

public:
  LivenessReport(mlir::ModuleOp &module) : module(module) {}

  std::string generateTextReport();
  std::string emitJSONReport();
  llvm::DenseMap<Value, std::vector<Operation *>> &getLiveness() {
    return livenessIntervals;
  };

private:
  void resolveLiveness();

  mlir::ModuleOp &module;
  llvm::DenseMap<Value, std::vector<Operation *>> livenessIntervals;
};

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif
