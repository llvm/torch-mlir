//===- mnist-playground.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include <torch/torch.h>

#include <chrono>

using namespace mlir;
using llvm::Error;
using llvm::ErrorOr;
using llvm::Expected;
using llvm::StringError;
using llvm::Twine;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

Expected<std::unique_ptr<refback::JITModule>>
createJITModule(std::string mlirFile, mlir::DialectRegistry &registry,
                ArrayRef<StringRef> sharedLibs, bool optimize) {
  MLIRContext context;
  registry.loadAll(&context);
  OwningModuleRef moduleRef = parseSourceFile(mlirFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + mlirFile);

  ModuleOp module = *moduleRef;

  // Compile.
  PassManager pm(module.getContext(), OpPassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);
  refback::JITModule::buildBackendCompilationPipeline(pm, optimize);
  if (failed(pm.run(module)))
    return make_string_error(Twine("error compiling to jit backend"));

  return refback::JITModule::fromCompiledModule(module, sharedLibs);
}

//===----------------------------------------------------------------------===//
// Benchmarking / correctness-testing code.
//===----------------------------------------------------------------------===//

static Expected<std::vector<at::Tensor>>
invokeJITModuleWithATenTensors(refback::JITModule &jitModule,
                               StringRef invokeFunction,
                               std::vector<at::Tensor> &args) {

  // Do a bit of checking. We don't handle all possible tensors right now.
  std::vector<at::TensorArg> tensorArgs;
  for (auto arg : llvm::enumerate(args))
    tensorArgs.push_back(at::TensorArg(arg.value(), "arg", arg.index()));
  at::CheckedFrom c = "converting to refbackrt::Tensor";
  for (auto &tensorArg : tensorArgs)
    at::checkScalarType(c, tensorArg, at::ScalarType::Float);
  at::checkAllContiguous(c, tensorArgs);

  SmallVector<refbackrt::Ref<refbackrt::Tensor>, 6> refbackInputs;
  for (at::Tensor arg : args) {
    SmallVector<int32_t, 6> extents(arg.sizes().begin(), arg.sizes().end());
    float *data = arg.storage().data<float>();
    // This does a deep copy of the data. Let's see if it shows up on the
    // profile.
    refbackInputs.push_back(refbackrt::Tensor::create(
        refbackrt::ArrayRef<int32_t>(extents.data(), extents.size()),
        refbackrt::ElementType::F32, data));
  }

  // Invoke the RefBackend function.
  auto expectedOutputs = jitModule.invoke(invokeFunction, refbackInputs);
  if (!expectedOutputs)
    return expectedOutputs.takeError();
  auto refbackrtOutputs = std::move(*expectedOutputs);

  std::vector<at::Tensor> results;
  for (auto output : refbackrtOutputs) {
    std::vector<int64_t> sizes(output->getExtents().data(),
                               output->getExtents().data() +
                                   output->getExtents().size());
    // Make a copy for passing to at::from_blob, which does its own internal
    // reference counting.
    auto *dataCopy = std::malloc(output->getDataByteSize());
    std::memcpy(dataCopy, output->getData(), output->getDataByteSize());
    results.push_back(at::from_blob(
        dataCopy, sizes, [](void *p) { std::free(p); }, at::kFloat));
  }
  return results;
}

using InvocationFunction =
    std::function<Expected<std::vector<at::Tensor>>(std::vector<at::Tensor>)>;

struct BenchmarkResult {
  int numRuns;
  float nsPerRun;
};

std::ostream &operator<<(std::ostream &os, const BenchmarkResult &result) {
  os << "numRuns: " << result.numRuns << " nsPerRun: " << std::scientific
     << result.nsPerRun << std::defaultfloat;
  return os;
}

Expected<BenchmarkResult> benchmark(std::function<Error()> f) {
  for (int itersAtATime = 1;; itersAtATime *= 2) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < itersAtATime; i++) {
      auto error = f();
      if (error)
        return std::move(error);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = end - start;

    // If the runtime is longer than 0.5 seconds, it's reliable enough.
    if (elapsed.count() > 0.5f) {
      BenchmarkResult result;
      result.numRuns = itersAtATime;
      result.nsPerRun = elapsed.count() * 10e9 / itersAtATime;
      return result;
    }
  }
  return make_string_error("too short running to benchmark!");
}

static Error doIt(InvocationFunction ptFunc, InvocationFunction refBackendFunc,
                  bool doBenchmark, int numCorrectnessTests) {

  torch::manual_seed(42);
  torch::set_num_threads(1);

  std::vector<at::Tensor> args;
  args.push_back(at::rand({784, 100}));
  args.push_back(at::rand({10, 784}));
  args.push_back(at::rand({10, 1}));

  // Initial correctness check of the two functions.
  for (int correctnessTest = 0; correctnessTest < numCorrectnessTests;
       correctnessTest++) {
    auto expectedPt = ptFunc(args);
    auto expectedRefBackend = refBackendFunc(args);
    if (!expectedPt)
      return expectedPt.takeError();
    if (!expectedRefBackend)
      return expectedRefBackend.takeError();
    auto pt = std::move(*expectedPt);
    auto refBackend = std::move(*expectedRefBackend);
    if (pt.size() != refBackend.size())
      return make_string_error("mismatch in result arity!");
    for (int i = 0, e = pt.size(); i < e; i++) {
      if (!at::allclose(pt[i], refBackend[i])) {
        std::cout << "PyTorch:\n" << pt[i] << "\n";
        std::cout << "RefBackend:\n" << refBackend[i] << "\n";
        return make_string_error(Twine("mismatch in result contents ") +
                                 Twine(i) + Twine(" on correctness test #") +
                                 Twine(correctnessTest));
      }
    }
  }
  if (!doBenchmark)
    return Error::success();

  // Benchmark the two against each other.
  BenchmarkResult ptBenchmarkResult;
  BenchmarkResult refBackendBenchmarkResult;
  {
    auto expectedResult =
        benchmark([&]() -> Error { return ptFunc(args).takeError(); });
    if (!expectedResult)
      return expectedResult.takeError();
    ptBenchmarkResult = std::move(*expectedResult);
  }

  {
    auto expectedResult =
        benchmark([&]() -> Error { return refBackendFunc(args).takeError(); });
    if (!expectedResult)
      return expectedResult.takeError();
    refBackendBenchmarkResult = std::move(*expectedResult);
  }
  std::cout << "PyTorch: " << ptBenchmarkResult << "\n";
  std::cout << "RefBackend: " << refBackendBenchmarkResult << "\n";
  std::cout << "Ratio (RefBackend / PyTorch): "
            << refBackendBenchmarkResult.nsPerRun / ptBenchmarkResult.nsPerRun
            << "\n";

  // TODO: Check for memory leaks?

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

  cl::list<std::string> sharedLibs{"shared-libs", cl::ZeroOrMore,
                                   cl::MiscFlags::CommaSeparated,
                                   cl::desc("Libraries to link dynamically")};
  cl::opt<bool> optimize{
      "optimize", cl::Optional,
      cl::desc("whether the refback pass pipeline should run optimizations"),
      cl::init(false)};

  cl::opt<bool> benchmark{"benchmark", cl::Optional,
                          cl::desc("whether to do a benchmark comparison"),
                          cl::init(true)};

  cl::opt<uint32_t> numCorrectnessTests{
      "num-correctness-tests", cl::Optional,
      cl::desc("how many correctness tests to run (useful for nondeterministic "
               "correctness failures"),
      cl::init(1)};
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
  llvm::cl::ParseCommandLineOptions(argc, argv, "mnist playground utility\n");

  SmallVector<StringRef, 6> sharedLibs(options.sharedLibs.begin(),
                                       options.sharedLibs.end());
  auto expectedJITModule = createJITModule(options.inputFile, registry,
                                           sharedLibs, options.optimize);
  if (Error error = expectedJITModule.takeError())
    llvm::report_fatal_error(llvm::toString(std::move(error)),
                             /*gen_crash_diag=*/false);
  auto jitModule = std::move(*expectedJITModule);

  Error error = doIt(
      [](std::vector<at::Tensor> args) {
        auto image = args[0];
        auto weights = args[1];
        auto biases = args[2];
        auto v0 = at::matmul(weights, image);
        auto v1 = at::add(v0, biases);
        return std::vector<at::Tensor>{v1};
      },
      [&](std::vector<at::Tensor> args) {
        return invokeJITModuleWithATenTensors(*jitModule,
                                              options.invokeFunction, args);
      },
      options.benchmark, options.numCorrectnessTests);

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
