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
      module(mlirModuleCreateEmpty(unknownLoc)), typeMapper(context) {
  // TODO: Rework this once dialect registration C-APIs are in place.
  // https://reviews.llvm.org/D88162
  mlirRegisterAllDialects(context);
  npcompRegisterAllDialects(context);

  // Terminator will always be the first op of an empty module.
  terminator = mlirBlockGetFirstOperation(getBodyBlock());
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
ModuleBuilder::startCaptureFunction(std::string &name,
                                    std::vector<at::Tensor> args) {
  // TODO: Verify that arguments do not alias each other.
  llvm::SmallVector<MlirType, 4> inputTypes;
  for (auto &arg : args) {
    inputTypes.push_back(typeMapper.forwardTensorToType(arg));
  }

  // TODO: Extract a traceback and use in place of unknownLoc.
  auto funcBuilder =
      FuncBuilder::createFunction(context, unknownLoc, name, inputTypes);
  mlirBlockInsertOwnedOperationBefore(getBodyBlock(), terminator,
                                      funcBuilder->getFuncOp());

  // Map block arguments.
  MlirBlock entryBlock = funcBuilder->getEntryBlock();
  assert(mlirBlockGetNumArguments(entryBlock) ==
             static_cast<intptr_t>(args.size()) &&
         "entry block incorrect arg arity");
  for (auto it : llvm::enumerate(args)) {
    funcBuilder->mapTensor(it.value(),
                           mlirBlockGetArgument(entryBlock, it.index()));
  }
  return std::make_shared<AcapController>(std::move(funcBuilder));
}

MlirBlock ModuleBuilder::getBodyBlock() {
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  return mlirRegionGetFirstBlock(mlirOperationGetRegion(moduleOp, 0));
}

void ModuleBuilder::bind(py::module &m) {
  py::class_<ModuleBuilder>(m, "ModuleBuilder")
      .def(py::init<>())
      .def("__str__", &ModuleBuilder::getAsm)
      .def("capture_function", &ModuleBuilder::startCaptureFunction,
           py::keep_alive<0, 1>());
}
