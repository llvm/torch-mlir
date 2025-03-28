//===- mlir_utils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_MLIR_UTILS_H
#define TORCHMLIRJITIRIMPORTER_CSRC_MLIR_UTILS_H

#include <cstring>
#include <string>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"

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
                                    const std::vector<MlirValue> &values) {
  mlirOperationStateAddOperands(&state, values.size(), values.data());
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    c10::ArrayRef<MlirValue> values) {
  mlirOperationStateAddOperands(&state, values.size(), values.data());
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    MlirType resultType) {
  mlirOperationStateAddResults(&state, 1, &resultType);
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    const std::vector<MlirType> &resultTypes) {
  if (resultTypes.empty())
    return; // must not proceed when resultTypes.data() is nullptr.
  mlirOperationStateAddResults(&state, resultTypes.size(), resultTypes.data());
}

inline void addToMlirOperationState(MlirOperationState &state,
                                    c10::ArrayRef<MlirType> resultTypes) {
  if (resultTypes.empty())
    return; // must not proceed when resultTypes.data() is nullptr.
  mlirOperationStateAddResults(&state, resultTypes.size(), resultTypes.data());
}

template <typename T>
void addToMlirOperationState(MlirOperationState &state, c10::optional<T> o) {
  if (o.has_value()) {
    addToMlirOperationState(state, o.value());
  }
}

inline void addToMlirOperationState(MlirOperationState &state) {}

template <typename T, typename U, typename... Ts>
void addToMlirOperationState(MlirOperationState &state, T &&t, U &&u,
                             Ts &&...ts) {
  addToMlirOperationState(state, std::forward<T>(t));
  addToMlirOperationState(state, std::forward<U>(u), std::forward<Ts>(ts)...);
}

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

#endif // TORCHMLIRJITIRIMPORTER_CSRC_MLIR_UTILS_H
