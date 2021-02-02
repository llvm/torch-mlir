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
#include <vector>

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

inline void addToMlirOperationState(MlirOperationState &state,
                                    MlirNamedAttribute namedAttr) {
  mlirOperationStateAddAttributes(&state, 1, &namedAttr);
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    MlirRegion region) {
  mlirOperationStateAddOwnedRegions(&state, 1, &region);
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    MlirValue value) {
  mlirOperationStateAddOperands(&state, 1, &value);
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    std::vector<MlirValue> &&values) {
  mlirOperationStateAddOperands(&state, values.size(), values.data());
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    MlirType resultType) {
  mlirOperationStateAddResults(&state, 1, &resultType);
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    std::vector<MlirType> &&resultTypes) {
  mlirOperationStateAddResults(&state, resultTypes.size(), resultTypes.data());
}

template <typename T, typename... Ts>
void addToMlirOperationState(MlirOperationState &state, T &&t, Ts &&...ts) {
  addToMlirOperationState(state, std::forward<T>(t));
  addToMlirOperationState(state, std::forward<Ts>(ts)...);
}

inline void addToMlirOperationState(MlirOperationState &state) {}

template <typename... Ts>
MlirOperation createMlirOperation(std::string name, MlirLocation loc,
                                  Ts &&...ts) {
  MlirOperationState state = mlirOperationStateGet(toMlirStringRef(name), loc);
  addToMlirOperationState(state, std::forward<Ts>(ts)...);
  return mlirOperationCreate(&state);
}

template <typename... Ts>
MlirOperation createMlirOperationAtEnd(MlirBlock block, std::string name,
                                       MlirLocation loc, Ts &&...ts) {
  MlirOperation operation =
      createMlirOperation(name, loc, std::forward<Ts>(ts)...);
  mlirBlockInsertOwnedOperationBefore(block, mlirBlockGetTerminator(block),
                                      operation);
  return operation;
}

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_MLIR_UTILS_H
