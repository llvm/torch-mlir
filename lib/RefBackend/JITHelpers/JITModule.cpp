//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/RefBackend/JITHelpers/JITModule.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "npcomp/RefBackend/RefBackend.h"

#include <sstream>

using namespace refback;
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
  NPCOMP::RefBackendLoweringPipelineOptions options;
  options.optimize = optimize;
  NPCOMP::createTCFRefBackendLoweringPipeline(pm, options);
}

llvm::Expected<std::unique_ptr<JITModule>>
JITModule::fromCompiledModule(mlir::ModuleOp module,
                              llvm::ArrayRef<llvm::StringRef> sharedLibs) {
  // Ensure LLVM Dialect -> LLVM IR translations are available.
  mlir::registerLLVMDialectTranslation(*module->getContext());
  // Build the JITModule.
  auto expectedEngine = ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr,
      /*transformer=*/[](llvm::Module *) { return Error::success(); },
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
      reinterpret_cast<refbackrt::ModuleDescriptor *>(*expectedAddress);
  return std::move(ret);
}

// Converter for bridging to refbackrt llvm-lookalike data structures.
static refbackrt::StringRef toRefbackrt(llvm::StringRef s) {
  return refbackrt::StringRef(s.data(), s.size());
}

template <typename T>
static refbackrt::ArrayRef<T> toRefbackrt(llvm::ArrayRef<T> a) {
  return refbackrt::ArrayRef<T>(a.data(), a.size());
}

template <typename T>
static refbackrt::MutableArrayRef<T> toRefbackrt(llvm::MutableArrayRef<T> a) {
  return refbackrt::MutableArrayRef<T>(a.data(), a.size());
}

static std::string stringifyShape(refbackrt::ArrayRef<std::int32_t> extents) {
  static constexpr char *kDynamicDimAsString = "?";
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < extents.size(); i++) {
    if (extents[i] < 0)
      ss << kDynamicDimAsString;
    else
      ss << extents[i];
    if (i != extents.size() - 1)
      ss << "x";
  }
  ss << ")";
  return ss.str();
}

llvm::Expected<llvm::SmallVector<refbackrt::RtValue, 6>>
JITModule::invoke(llvm::StringRef functionName,
                  llvm::ArrayRef<refbackrt::RtValue> inputs) {
  refbackrt::FunctionMetadata metadata;
  if (refbackrt::failed(refbackrt::getMetadata(
          descriptor, toRefbackrt(functionName), metadata)))
    return make_string_error("unknown function: " + Twine(functionName));
  SmallVector<refbackrt::RtValue, 6> outputs(metadata.numOutputs);
  if (metadata.numInputs != static_cast<std::int32_t>(inputs.size()))
    return make_string_error("invoking '" + Twine(functionName) +
                             "': expected " + Twine(metadata.numInputs) +
                             " inputs");

  // Verify user input types and shapes match what the compiler expects
  for (int i = 0; i < metadata.numInputs; i++) {
    auto &input = inputs[i];
    auto &inputArgInfo = metadata.inputArgInfos[i];
    if (refbackrt::failed(checkRtValueArgTypes(input, inputArgInfo)))
      return make_string_error(
          "invoking '" + Twine(functionName) +
          "': input argument type mismatch. actual (provided by user): " +
          Twine(inputs[i].tagKind().str()) + ", expected (from compiler): " +
          Twine(getArgTypeAsStringRef(inputArgInfo.argType).str()));
    if (refbackrt::failed(checkRtValueShapes(input, inputArgInfo)))
      return make_string_error(
          "invoking '" + Twine(functionName) + "': input shape mismatch (%arg" +
          Twine(i) + "). " + "actual (provided by user): " +
          stringifyShape(input.toTensor()->getExtents()) +
          ", expected (from compiler): " +
          stringifyShape(refbackrt::ArrayRef<int32_t>(
              inputArgInfo.extents.data(), inputArgInfo.rank)));
  }

  // Create the correct output RtValue based on FuncMetadata,
  // which contains the arg types (scalar, Tensor, etc.), element types (only
  // applicable if not scalar) and shapes (also only applicable if not scalar)
  //
  // Currently we have to give each RtValue an output type so that we know
  // how to pack / unpack the outputs properly across the ABI boundary in
  // refbackrt::invoke. As a result, we can't just rely on the default
  // construction of each output argument type (otherwise RtValue will have
  // Tag::kNone) currently without passing the ArgInfo structs down to the
  // Runtime level, so we deal with the output type creation here.
  for (int i = 0; i < metadata.numOutputs; i++) {
    outputs[i] = std::move(
        refbackrt::createRtValueFromOutputArgInfo(metadata.outputArgInfos[i]));
  }

  refbackrt::invoke(
      descriptor, toRefbackrt(functionName), toRefbackrt(inputs),
      toRefbackrt(llvm::makeMutableArrayRef(outputs.data(), outputs.size())));
  return outputs;
}
