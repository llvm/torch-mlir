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
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "npcomp/E2E/E2E.h"
#include "npcomp/InitAll.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;
using llvm::Error;
using llvm::StringError;
using llvm::Twine;

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

// TODO: This will all go away once we refactor a cleanly defined "runtime"
// layer for npcomp.
// This is a toy anyway, because our current ABI with memref descriptors
// doesn't even support correctly returning owned tensor values.

namespace {
// Helper for converting a DenseFPElementsAttr into the data structures
// needed for passing into the runtime.
//
// For now, we don't even worry about freeing memory, since this will be
// soon superceded by a simple runtime.
struct RankAndDescriptor {
  RankAndDescriptor(DenseFPElementsAttr attr) {
    SmallVector<float, 10> elements;
    for (APFloat element : attr)
      elements.push_back(element.convertToFloat());
    auto type = attr.getType().cast<ShapedType>();
    assert(type.getElementType().isF32() && "only handle f32 right now");
    assert(type.getRank() == 1 && "only handle rank 1 now");
    if (type.getRank() == 1) {
      rank = 1;
      auto descriptor = new StridedMemRefType<float, 1>;
      descriptor->basePtr = new float[elements.size()];
      descriptor->data = descriptor->basePtr;
      descriptor->offset = 0;
      descriptor->sizes[0] = elements.size();
      descriptor->strides[0] = 1;
      descriptorVoidPtr = static_cast<void *>(descriptor);
      return;
    }
    llvm::report_fatal_error("could not create RankAndDescriptor");
  }

  int64_t rank;
  void *descriptorVoidPtr;
};
} // namespace

namespace {
// Prepares the data from a set of attributes for passing to
// mlir::ExecutionEngine::invoke, according to the ABI of the npcomp
// runtime.
//
// This class mostly exists to own the data for the descriptors and make
// sure they are cleaned up properly.
class InvocationArgs {
public:
  static llvm::Expected<InvocationArgs>
  fromAttributes(ArrayRef<Attribute> attrs) {
    InvocationArgs result;
    for (auto attr : attrs) {
      auto denseElements = attr.dyn_cast<DenseFPElementsAttr>();
      if (!denseElements || !denseElements.getType().getElementType().isF32())
        return make_string_error("only support f32 for now");
      result.descriptors.push_back(RankAndDescriptor(denseElements));
    }
    for (auto descriptor : result.descriptors) {
      result.packedArgs.push_back(static_cast<void *>(&descriptor.rank));
      result.packedArgs.push_back(static_cast<void *>(&descriptor.descriptorVoidPtr));
    }
    return result;
  }

  // Get packed args in a form suitable for passing to
  // mlir::ExecutionEngine::invoke.
  MutableArrayRef<void *> getPackedArgs() { return packedArgs; }

private:
  SmallVector<RankAndDescriptor, 6> descriptors;
  SmallVector<void *, 6> packedArgs;
};
} // namespace

Error compileAndRun(std::string mlirFile, std::string invokeFunction,
                    ArrayRef<StringRef> argValues,
                    ArrayRef<StringRef> sharedLibs) {
  MLIRContext context;
  OwningModuleRef moduleRef = parseSourceFile(mlirFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + mlirFile);
  ModuleOp module = *moduleRef;

  SymbolTable symbolTable(module);
  FuncOp func = dyn_cast_or_null<FuncOp>(symbolTable.lookup(invokeFunction));
  if (!func) {
    return make_string_error(Twine("could not find function: ") +
                             invokeFunction);
  }
  if (func.getType().getInputs().size() != argValues.size()) {
    return make_string_error(Twine("mismatch between number of --arg-value's "
                                   "and number of expected arguments (") +
                             Twine(argValues.size()) + " vs " +
                             Twine(func.getType().getInputs().size()) + ")");
  }
  SmallVector<Attribute, 6> args;
  for (auto t : llvm::zip(argValues, func.getType().getInputs())) {
    auto attr = parseAttribute(std::get<0>(t), &context);
    if (!attr)
      return make_string_error(Twine("could not parse arg value: ") +
                               std::get<0>(t));
    if (failed(verifyCompatibleShape(attr.getType(), std::get<1>(t))))
      return make_string_error(Twine("incompatible shape for arg value: ") +
                               std::get<0>(t));
    args.push_back(attr);
  }

  // Run the lowering.
  PassManager pm(&context, /*verifyPasses=*/true);
  applyPassManagerCLOptions(pm);

  NPCOMP::createE2ELoweringPipeline(pm);
  llvm::errs() << "RUNNING PIPELINE: ";
  pm.printAsTextualPipeline(llvm::errs());
  llvm::errs() << "\n";

  if (failed(pm.run(module)))
    return make_string_error("could not lower module");
  llvm::outs() << "FINAL MODULE\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  auto expectedEngine = ExecutionEngine::create(
      module, [](llvm::Module *) {
    return Error::success(); },
      /*jitCodeGenOptLevel=*/llvm::None, llvm::to_vector<6>(sharedLibs));
  if (!expectedEngine)
    return expectedEngine.takeError();
  auto engine = std::move(*expectedEngine);

  auto expectedInvocationArgs = InvocationArgs::fromAttributes(args);
  if (!expectedInvocationArgs)
    return expectedInvocationArgs.takeError();
  auto error = engine->invoke(invokeFunction,
                              expectedInvocationArgs->getPackedArgs());
  if (error)
    return error;

  llvm::errs() << "SUCCESS\n";
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Main-related init and option parsing.
//===----------------------------------------------------------------------===//

namespace {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{"input", cl::Required,
                                 cl::desc("the input .mlir file")};
  cl::opt<std::string> invokeFunction{"invoke", cl::Required,
                                      cl::desc("function to invoke")};
  cl::list<std::string> argValues{"arg-value", cl::ZeroOrMore,
                                  cl::desc("Arguments to the called function")};

  cl::list<std::string> sharedLibs{"shared-libs", cl::ZeroOrMore,
                                   cl::MiscFlags::CommaSeparated,
                                   cl::desc("Libraries to link dynamically")};
};
} // namespace

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  mlir::NPCOMP::registerAllDialects();
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
  Error error = compileAndRun(options.inputFile, options.invokeFunction,
                              argValues, sharedLibs);

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
