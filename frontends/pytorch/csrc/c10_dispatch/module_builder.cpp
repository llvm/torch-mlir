//===- module_builder.cpp -------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "module_builder.h"

#include "mlir-c/Registration.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Registration.h"

namespace py = pybind11;
using namespace torch_mlir;

namespace {
/// Accumulates into a python string from a method that accepts an
/// MlirStringCallback.
/// TODO: Remove this once the MLIR Python objects are exposed directly.
struct PyPrintAccumulator {
  py::list parts;

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](const char *part, intptr_t size, void *userData) {
      PyPrintAccumulator *printAccum =
          static_cast<PyPrintAccumulator *>(userData);
      py::str pyPart(part, size); // Decodes as UTF-8 by default.
      printAccum->parts.append(std::move(pyPart));
    };
  }

  py::str join() {
    py::str delim("", 0);
    return delim.attr("join")(parts);
  }
};
} // namespace

ModuleBuilder::ModuleBuilder()
    // TODO: Once the MLIR C/Python capsule API is in place, these should be
    // derived from Python level objects (which will provide better lifetime
    // semantics and interop). Until then, they are just scoped to this instance
    // and must not escape.
    : context(mlirContextCreate()), unknownLoc(mlirLocationUnknownGet(context)),
      module(mlirModuleCreateEmpty(unknownLoc)) {
  // TODO: Rework this once dialect registration C-APIs are in place.
  // https://reviews.llvm.org/D88162
  mlirRegisterAllDialects(context);
  npcompRegisterAllDialects(context);
}

ModuleBuilder::~ModuleBuilder() {
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
}

py::str ModuleBuilder::getAsm() {
  MlirOperation operation = mlirModuleGetOperation(module);
  PyPrintAccumulator printAccum;
  mlirOperationPrint(operation, printAccum.getCallback(),
                     printAccum.getUserData());
  return printAccum.join();
}

std::shared_ptr<AcapController>
ModuleBuilder::startCaptureFunction(std::string &name) {
  // TODO: Populate input/result types.
  llvm::SmallVector<MlirType, 4> inputTypes;
  llvm::SmallVector<MlirType, 4> resultTypes;
  MlirOperation funcOp = createFunction(name, inputTypes, resultTypes);
  return std::make_shared<AcapController>(funcOp);
}

// TODO: Implement an mlir-c API for creating a function and avoid the danger
// of getting the below wrong.
MlirOperation
ModuleBuilder::createFunction(std::string &name,
                              llvm::SmallVectorImpl<MlirType> &inputTypes,
                              llvm::SmallVectorImpl<MlirType> &resultTypes) {
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  MlirBlock moduleBlock =
      mlirRegionGetFirstBlock(mlirOperationGetRegion(moduleOp, 0));

  llvm::SmallVector<MlirNamedAttribute, 4> funcAttrs;
  funcAttrs.push_back(mlirNamedAttributeGet(
      "type", mlirTypeAttrGet(mlirFunctionTypeGet(
                  context, inputTypes.size(), inputTypes.data(),
                  resultTypes.size(), resultTypes.data()))));
  funcAttrs.push_back(mlirNamedAttributeGet(
      "sym_name", mlirStringAttrGet(context, name.size(), name.data())));

  // TODO: Extract current traceback and use it for location.
  MlirOperationState state = mlirOperationStateGet("func", unknownLoc);
  mlirOperationStateAddAttributes(&state, funcAttrs.size(), funcAttrs.data());
  {
    // Don't access these once ownership transferred.
    MlirRegion bodyRegion = mlirRegionCreate();
    MlirBlock entryBlock =
        mlirBlockCreate(inputTypes.size(), inputTypes.data());
    mlirRegionInsertOwnedBlockAfter(bodyRegion, {nullptr}, entryBlock);
    mlirOperationStateAddOwnedRegions(&state, 1, &bodyRegion);
  }

  MlirOperation funcOp = mlirOperationCreate(&state);
  mlirBlockInsertOwnedOperationAfter(moduleBlock, {nullptr}, funcOp);
  return funcOp;
}

void ModuleBuilder::bind(py::module &m) {
  py::class_<ModuleBuilder>(m, "ModuleBuilder")
      .def(py::init<>())
      .def("__str__", &ModuleBuilder::getAsm)
      .def("capture_function", &ModuleBuilder::startCaptureFunction,
           py::keep_alive<0, 1>());
}
