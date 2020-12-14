//===- mlir_utils.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_MLIR_UTILS_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_MLIR_UTILS_H

#include <cstring>
#include <string>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

namespace torch_mlir {

inline MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

inline MlirStringRef toMlirStringRef(const char *s) {
  return mlirStringRefCreate(s, std::strlen(s));
}

inline MlirNamedAttribute toMlirNamedAttribute(const char *s,
                                               MlirAttribute attr) {
  MlirContext context = mlirAttributeGetContext(attr);
  MlirIdentifier ident = mlirIdentifierGet(context, toMlirStringRef(s));
  return mlirNamedAttributeGet(ident, attr);
}

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_MLIR_UTILS_H
