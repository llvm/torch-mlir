//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_TRANSFORMS_REIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_TRANSFORMS_REIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace mlir {
namespace torch {
namespace Torch {

enum class LibraryFunctionKind {
  ShapeFunction,
  DtypeFunction,
  Decomposition,
  HasValueSemantics
};

// Searches the function library for an abstract interpretation function for
// `op`. If one is found, wraps the op in a `CalculateOp`, with the op placed in
// the first region, and a call to the abstract interpretation function is
// inserted into the second region.
//
// Note: this returns success if no abstract interpretation function is found,
// since some abstract interpretation functions (such as decompositions) are
// optional.
//
// Note: This function does *not* import the abstract interpretation function
// from the library into the IR.
LogicalResult wrapWithCalculateOpIfLibraryFunctionAvailable(
    Operation *op, ModuleOp library, LibraryFunctionKind funcKind,
    SmallVector<std::string> &libFuncNamesUsed,
    function_ref<FailureOr<SmallVector<Value>>(OpBuilder &, Location,
                                               ValueRange, func::FuncOp)>
        libFuncArgsBuilder);

// Imports the functions in `functionsNeeded` from the library into the module.
// This function assumes that all functions needed exist in the library.
//
// Note: This function modifies the library.
void importLibraryFunctions(ModuleOp module, ModuleOp library,
                            SmallVector<std::string> functionsNeeded);

// Recursively adjust `operand` to match `desiredType`.
//
// This function by default handles a few types such as `UnionType`,
// `OptionalType`, and `ListType`, to name a few. Handling of base element types
// can be customized by defining `baseTransformation`, which gets called at the
// beginning of each recursive call. This function can be thought of as mapping
// `baseTransformation` across `UnionType/OptionalType/ListType`.
FailureOr<Value> adjustFunctionArg(
    OpBuilder &b, Location loc, Value operand, Type desiredType,
    function_ref<Value(OpBuilder &, Location, Value, Type)> baseTransformation =
        [](OpBuilder &, Location, Value operand, Type) { return operand; });

std::string getLibraryFunctionPrefix(LibraryFunctionKind libFuncKind);

// Parse MLIR module at `filename` into a ModuleOp that will then
// be appended to an existing, fully hydrated, ModuleOp; note the module
// should have been instantiated with an associated context like so:
// `OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));`
LogicalResult loadExtraLibrary(const std::string &filename,
                               OwningOpRef<ModuleOp> &moduleToAppendTo);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_REIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H
