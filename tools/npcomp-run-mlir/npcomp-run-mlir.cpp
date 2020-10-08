//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility binary for compiling and running code through the npcomp
// compiler/runtime stack.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "npcomp/InitAll.h"
#include "npcomp/RefBackend/JITHelpers/JITModule.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

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

static Expected<refbackrt::Ref<refbackrt::Tensor>>
convertAttrToTensor(Attribute attr) {
  auto type = attr.getType().dyn_cast<RankedTensorType>();
  if (!type)
    return make_string_error("unhandled argument type; must be a tensor type");
  auto extents = llvm::to_vector<6>(llvm::map_range(
      type.getShape(), [](int64_t x) { return static_cast<std::int32_t>(x); }));
  auto elementType = type.getElementType();
  if (auto denseFp = attr.dyn_cast<DenseFPElementsAttr>()) {
    if (elementType.isF32()) {
      auto values = llvm::to_vector<100>(llvm::map_range(
          denseFp, [](APFloat f) { return f.convertToFloat(); }));
      return refbackrt::Tensor::create(
          refbackrt::ArrayRef<std::int32_t>(extents.data(), extents.size()),
          refbackrt::ElementType::F32, static_cast<void *>(values.data()));
    }
  }
  return make_string_error("unhandled argument");
}

static Expected<SmallVector<refbackrt::Ref<refbackrt::Tensor>, 6>>
createInputs(ArrayRef<StringRef> argValues) {
  MLIRContext context;
  SmallVector<refbackrt::Ref<refbackrt::Tensor>, 6> ret;
  for (auto argValue : argValues) {
    auto attr = parseAttribute(argValue, &context);
    if (!attr)
      return make_string_error(Twine("could not parse arg value: ") + argValue);
    auto expectedTensor = convertAttrToTensor(attr);
    if (!expectedTensor)
      return expectedTensor.takeError();
    ret.push_back(std::move(*expectedTensor));
  }
  return ret;
}

static Type convertToMLIRType(refbackrt::ElementType type, Builder &builder) {
  switch (type) {
  case refbackrt::ElementType::F32:
    return builder.getF32Type();
  }
}

static RankedTensorType getCorrespondingMLIRTensorType(refbackrt::Tensor &tensor,
                                                       Builder &builder) {
  auto elementType = convertToMLIRType(tensor.getElementType(), builder);
  SmallVector<int64_t, 6> extents;
  for (int i = 0, e = tensor.getRank(); i < e; i++)
    extents.push_back(tensor.getExtent(i));
  return RankedTensorType::get(extents, elementType);
}

static Attribute convertToMLIRAttribute(refbackrt::Tensor &tensor,
                                        Builder &builder) {
  RankedTensorType type = getCorrespondingMLIRTensorType(tensor, builder);
  switch (tensor.getElementType()) {
  case refbackrt::ElementType::F32: {
    SmallVector<float, 100> values;
    auto *basePtr = tensor.getData<float>();
    for (int i = 0, e = type.getNumElements(); i < e; i++)
      values.push_back(basePtr[i]);
    return DenseFPElementsAttr::get(type, values);
  }
  }
}

static void printOutput(refbackrt::Tensor &tensor, llvm::raw_ostream &os) {
  MLIRContext context;
  Builder builder(&context);
  auto attr = convertToMLIRAttribute(tensor, builder);
  attr.print(os);
}

static void printOutputs(ArrayRef<refbackrt::Ref<refbackrt::Tensor>> outputs,
                         llvm::raw_ostream &os) {
  for (auto output : llvm::enumerate(outputs)) {
    os << "output #" << output.index() << ": ";
    printOutput(*output.value(), os);
    os << "\n";
  }
}

Error compileAndRun(std::string mlirFile, mlir::DialectRegistry &registry,
                    std::string invokeFunction, ArrayRef<StringRef> argValues,
                    ArrayRef<StringRef> sharedLibs, bool optimize) {
  MLIRContext context;
  registry.loadAll(&context);
  OwningModuleRef moduleRef = parseSourceFile(mlirFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + mlirFile);

  ModuleOp module = *moduleRef;

  // Compile.
  PassManager pm(module.getContext(), /*verifyPasses=*/true);
  applyPassManagerCLOptions(pm);
  npcomp::JITModule::buildBackendCompilationPipeline(pm, optimize);
  if (failed(pm.run(module))) {
    return make_string_error(Twine("error compiling to jit backend"));
  }

  auto expectedJitModule =
      npcomp::JITModule::fromCompiledModule(module, sharedLibs);
  if (!expectedJitModule)
    return expectedJitModule.takeError();
  auto jitModule = std::move(*expectedJitModule);

  auto expectedInputs = createInputs(argValues);
  if (!expectedInputs)
    return expectedInputs.takeError();
  auto expectedOutputs = jitModule->invoke(invokeFunction, *expectedInputs);
  if (!expectedOutputs)
    return expectedOutputs.takeError();
  auto outputs = std::move(*expectedOutputs);
  printOutputs(outputs, llvm::outs());
  llvm::outs() << "SUCCESS\n";
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Main-related init and option parsing.
//===----------------------------------------------------------------------===//

namespace {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{
      cl::Positional, cl::desc("the input .mlir file"), cl::init("-")};
  cl::opt<std::string> invokeFunction{"invoke", cl::Required,
                                      cl::desc("function to invoke")};
  cl::list<std::string> argValues{"arg-value", cl::ZeroOrMore,
                                  cl::desc("Arguments to the called function")};

  cl::list<std::string> sharedLibs{"shared-libs", cl::ZeroOrMore,
                                   cl::MiscFlags::CommaSeparated,
                                   cl::desc("Libraries to link dynamically")};
  cl::opt<bool> optimize{
      "optimize", cl::Optional,
      cl::desc("whether the refback pass pipeline should run optimizations"),
      cl::init(false)};
};
} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::NPCOMP::registerAllDialects(registry);
  mlir::NPCOMP::registerAllPasses();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "npcomp compile+run utility\n");

  SmallVector<StringRef, 6> sharedLibs(options.sharedLibs.begin(),
                                       options.sharedLibs.end());
  SmallVector<StringRef, 6> argValues(options.argValues.begin(),
                                      options.argValues.end());
  Error error =
      compileAndRun(options.inputFile, registry, options.invokeFunction,
                    argValues, sharedLibs, options.optimize);

  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error),
                        [&exitCode](const llvm::ErrorInfoBase &info) {
                          llvm::errs() << "Error: ";
                          info.log(llvm::errs());
                          llvm::errs() << '\n';
                          exitCode = EXIT_FAILURE;
                        });
  return exitCode;
}
