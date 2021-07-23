//===- RefJITBackend.cpp - CAPI for RefJit --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/RefJITBackend.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/PassManager.h"
#include "npcomp/RefBackend/JITHelpers/JITModule.h"
#include "llvm/ADT/Optional.h"

using namespace llvm;
using namespace mlir;
using namespace refback;
using namespace refbackrt;

using ValueListCpp = SmallVector<RtValue, 4>;
DEFINE_C_API_PTR_METHODS(NpcompRefJitModule, JITModule)
DEFINE_C_API_PTR_METHODS(NpcompRefJitValueList, ValueListCpp)

static_assert(static_cast<int>(ElementType::F32) == NPCOMP_REFJIT_F32,
              "mismatched F32 mapping");

namespace {
template <typename T>
static Optional<T> checkError(llvm::Expected<T> &&expected,
                              char **errorMessageCstr, Twine banner = {}) {
  if (LLVM_LIKELY(expected))
    return std::move(*expected);

  std::string errorMessage;
  llvm::raw_string_ostream os(errorMessage);
  llvm::logAllUnhandledErrors(expected.takeError(), os, banner);
  os.flush();
  *errorMessageCstr = strdup(errorMessage.c_str());
  return llvm::None;
}

} // namespace

void npcompRefJitBuildBackendCompilationPipeline(MlirPassManager passManager,
                                                 bool optimize) {
  JITModule::buildBackendCompilationPipeline(*unwrap(passManager), optimize);
}

NpcompRefJitModule npcompRefJitModuleCreate(MlirModule moduleOp,
                                            MlirStringRef *sharedLibs,
                                            intptr_t sharedLibsSize,
                                            char **errorMessage) {
  SmallVector<llvm::StringRef> sharedLibsCpp;
  for (intptr_t i = 0; i < sharedLibsSize; ++i) {
    sharedLibsCpp.push_back(
        llvm::StringRef(sharedLibs[i].data, sharedLibs[i].length));
  }

  auto refJitModuleCpp =
      checkError(JITModule::fromCompiledModule(unwrap(moduleOp), sharedLibsCpp),
                 errorMessage, "error creating refjit module");
  if (!refJitModuleCpp)
    return {nullptr};
  return wrap(refJitModuleCpp->release());
}

void npcompRefJitModuleDestroy(NpcompRefJitModule module) {
  delete unwrap(module);
}

bool npcompRefJitModuleInvoke(NpcompRefJitModule m, MlirStringRef functionName,
                              NpcompRefJitValueList inputOutputs,
                              char **errorMessage) {
  ValueListCpp *ioList = unwrap(inputOutputs);
  auto results = checkError(
      unwrap(m)->invoke(llvm::StringRef(functionName.data, functionName.length),
                        *ioList),
      errorMessage, "error invoking function");
  ioList->clear();
  if (!results)
    return false;

  for (int i = 0, e = results->size(); i < e; ++i) {
    ioList->push_back(std::move((*results)[i]));
  }
  return true;
}

NpcompRefJitValueList npcompRefJitValueListCreate() {
  return wrap(new ValueListCpp());
}

void npcompRefJitValueListDestroy(NpcompRefJitValueList list) {
  delete unwrap(list);
}

intptr_t npcompRefJitValueListSize(NpcompRefJitValueList list) {
  return unwrap(list)->size();
}

void npcompRefJitValueAddTensorCopy(NpcompRefJitValueList list,
                                    NpcompRefJitElementType elementType,
                                    const int32_t *extents,
                                    intptr_t extentsSize, const void *data) {
  ElementType elementTypeCpp = static_cast<ElementType>(elementType);
  auto tensor =
      Tensor::create(refbackrt::ArrayRef<std::int32_t>(extents, extentsSize),
                     elementTypeCpp, const_cast<void *>(data));
  unwrap(list)->push_back(std::move(tensor));
}

bool npcompRefJitValueIsaTensor(NpcompRefJitValueList list, intptr_t i) {
  return (*unwrap(list))[i].isTensor();
}

void *npcompRefJitValueGetTensor(NpcompRefJitValueList list, intptr_t i,
                                 NpcompRefJitElementType *elementType,
                                 intptr_t *rank, const int32_t **extents) {
  auto tensor = (*unwrap(list))[i].toTensor();
  *elementType = static_cast<NpcompRefJitElementType>(tensor->getElementType());
  *rank = tensor->getRank();
  *extents = tensor->getExtents().data();
  return tensor->getData();
}
