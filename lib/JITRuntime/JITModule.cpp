//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/JITRuntime/JITModule.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "npcomp/E2E/E2E.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace npcomp;
using namespace mlir;
using llvm::Error;
using llvm::Expected;
using llvm::StringError;
using llvm::Twine;

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

JITModule::JITModule() {}

void JITModule::buildBackendCompilationPipeline(PassManager &pm,
                                                bool optimize) {
  NPCOMP::E2ELoweringPipelineOptions options;
  options.optimize = optimize;
  NPCOMP::createE2ELoweringPipeline(pm, options);
}

llvm::Expected<std::unique_ptr<JITModule>>
JITModule::fromCompiledModule(mlir::ModuleOp module,
                              llvm::ArrayRef<llvm::StringRef> sharedLibs) {
  auto expectedEngine = ExecutionEngine::create(
      module, [](llvm::Module *) { return Error::success(); },
      /*jitCodeGenOptLevel=*/llvm::None, llvm::to_vector<6>(sharedLibs));
  if (!expectedEngine)
    return expectedEngine.takeError();
  std::unique_ptr<JITModule> ret(new JITModule);
  ret->engine = std::move(*expectedEngine);
  // Here we abuse mlir::ExecutionEngine a bit. It technically returns a
  // function pointer, but here we look up a module descriptor.
  auto expectedAddress = ret->engine->lookup("__npcomp_module_descriptor");
  if (!expectedAddress)
    return expectedAddress.takeError();
  ret->descriptor =
      reinterpret_cast<npcomprt::ModuleDescriptor *>(*expectedAddress);
  return std::move(ret);
}

// Converter for bridging to npcomprt llvm-lookalike data structures.
static npcomprt::StringRef toNpcomprt(llvm::StringRef s) {
  return npcomprt::StringRef(s.data(), s.size());
}

template <typename T>
static npcomprt::ArrayRef<T> toNpcomprt(llvm::ArrayRef<T> a) {
  return npcomprt::ArrayRef<T>(a.data(), a.size());
}

template <typename T>
static npcomprt::MutableArrayRef<T> toNpcomprt(llvm::MutableArrayRef<T> a) {
  return npcomprt::MutableArrayRef<T>(a.data(), a.size());
}

llvm::Expected<llvm::SmallVector<npcomprt::Ref<npcomprt::Tensor>, 6>>
JITModule::invoke(llvm::StringRef functionName,
                  llvm::ArrayRef<npcomprt::Ref<npcomprt::Tensor>> inputs) {
  npcomprt::FunctionMetadata metadata;
  if (npcomprt::failed(npcomprt::getMetadata(
          descriptor, toNpcomprt(functionName), metadata)))
    return make_string_error("unknown function: " + Twine(functionName));
  SmallVector<npcomprt::Ref<npcomprt::Tensor>, 6> outputs(metadata.numOutputs);
  if (metadata.numInputs != static_cast<std::int32_t>(inputs.size()))
    return make_string_error("invoking '" + Twine(functionName) +
                             "': expected " + Twine(metadata.numInputs) +
                             " inputs");
  npcomprt::invoke(
      descriptor, toNpcomprt(functionName), toNpcomprt(inputs),
      toNpcomprt(llvm::makeMutableArrayRef(outputs.data(), outputs.size())));
  return outputs;
}
