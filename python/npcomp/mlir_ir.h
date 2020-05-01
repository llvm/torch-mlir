//===- mlir_ir.h - MLIR IR Bindings -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_MLIR_IR_H
#define NPCOMP_PYTHON_MLIR_IR_H

#include "pybind_utils.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

namespace mlir {

struct PyContext;

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

template <typename ListTy, typename ItemWrapperTy> class PyIpListWrapper {
public:
  using ThisTy = PyIpListWrapper<ListTy, ItemWrapperTy>;
  static void bind(py::module m, const char *className);
  PyIpListWrapper(ListTy &list) : list(list) {}

private:
  ListTy &list;
};

//===----------------------------------------------------------------------===//
// Wrapper types
//===----------------------------------------------------------------------===//

/// Wrapper around an Operation*.
struct PyBaseOperation {
  virtual ~PyBaseOperation();
  static void bind(py::module m);
  virtual Operation *getOperation() = 0;
};

/// Wrapper around Module, capturing a PyContext reference.
struct PyModuleOp : PyBaseOperation {
  PyModuleOp(std::shared_ptr<PyContext> context, ModuleOp moduleOp)
      : context(context), moduleOp(moduleOp) {}
  ~PyModuleOp();
  static void bind(py::module m);
  Operation *getOperation() override;
  std::string toAsm(bool enableDebugInfo, bool prettyForm,
                    int64_t largeElementLimit);

  std::shared_ptr<PyContext> context;
  ModuleOp moduleOp;
};

/// Wrapper around an Operation*.
struct PyOperationRef : PyBaseOperation {
  PyOperationRef(Operation *operation) : operation(operation) {}
  PyOperationRef(Operation &operation) : operation(&operation) {}
  ~PyOperationRef();
  static void bind(py::module m);
  Operation *getOperation() override;

  Operation *operation;
};

/// Wrapper around MLIRContext.
struct PyContext : std::enable_shared_from_this<PyContext> {
  static void bind(py::module m);
  PyModuleOp parseAsm(const std::string &asm_text);
  MLIRContext context;
};

/// Wrapper around a Block&.
struct PyBlockRef {
  PyBlockRef(Block &block) : block(block) {}
  static void bind(py::module m);
  Block &block;
};

/// Wrapper around a Region&.
struct PyRegionRef {
  PyRegionRef(Region &region) : region(region) {}
  static void bind(py::module m);
  Region &region;
};

/// Wrapper around an OpBuilder reference.
/// This class is inherently dangerous because it does not track ownership
/// of IR objects that it may be operating on and incorrect usage can cause
/// memory access errors, just as it can in C++. It is intended for use by
/// higher level constructs that are specifically coded to satisfy object
/// lifetime needs.
class PyBaseOpBuilder {
public:
  virtual ~PyBaseOpBuilder();
  static void bind(py::module m);
  virtual OpBuilder &getBuilder() = 0;
};

/// Wrapper around an instance of an OpBuilder.
class PyOpBuilder : public PyBaseOpBuilder {
public:
  PyOpBuilder(PyContext &context) : builder(&context.context) {}
  ~PyOpBuilder() override;
  static void bind(py::module m);
  OpBuilder &getBuilder() override;

private:
  OpBuilder builder;
};

} // namespace mlir

#endif // NPCOMP_PYTHON_MLIR_IR_H
