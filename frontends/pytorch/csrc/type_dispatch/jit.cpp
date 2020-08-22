//===- jit.cpp --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This file drives the generation and lowering of MLIR, followed by JIT
// compiling the resulting LLVM dialect.

#include "npcomp/Dialect/ATen/ATenDialect.h"
#include "npcomp/Dialect/ATen/ATenPasses.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>

#include "ATen/ArrayRef.h"
namespace at {
template <typename T> using ArrayRef = c10::ArrayRef<T>;
}
#include "ATen/Tensor.h"
#include <ATen/CPUType.h>

#include "jit.h"
#include "mlir_gen.h"
#include "tensor.h"
#include "torch_util.h"

#define DEBUG_TYPE "torch_mlir"

using namespace mlir;

namespace torch_mlir {

namespace {

int LowerATenDialect(mlir::ModuleOp module) {
  PassManager pm0(module.getContext());
  pm0.addPass(mlir::createCSEPass());

  // Lower to function calls.
  pm0.addPass(mlir::NPCOMP::aten::createATenLoweringPass());
  pm0.addPass(mlir::NPCOMP::aten::createReturnEliminationPass());

  if (failed(pm0.run(module))) {
    llvm::errs() << "aten to loops conversion failed ";
    return 1;
  }

  PassManager pm1(module.getContext());
  pm1.addPass(mlir::createLowerAffinePass());
  pm1.addPass(mlir::createLowerToCFGPass());
  pm1.addPass(mlir::createCSEPass());

  if (failed(pm1.run(module))) {
    llvm::errs() << "loops to std conversion failed ";
    return 1;
  }

  return 0;
}

int LowerStdDialect(mlir::ModuleOp module) {
  PassManager pm(module.getContext());

  struct LowerToLLVMOptions options;
  options.emitCWrappers = true;
  LLVM_DEBUG(module.print(llvm::outs()));

  pm.addPass(mlir::createLowerToLLVMPass(options));
  pm.addPass(mlir::createCSEPass());

  LLVM_DEBUG(module.print(llvm::outs()));

  if (failed(pm.run(module))) {
    llvm::errs() << "std to llvm conversion failed ";
    return 1;
  }

  if (!module)
    return 1;
  return 0;
}

template <typename T, int N> struct llvm_tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];
};

template <typename T, int N> void *setupArg(at::Tensor &t) {
  llvm_tensor_t<T, N> *arg = new llvm_tensor_t<T, N>;
  llvm_tensor_t<T, N> **arg_storage = new llvm_tensor_t<T, N> *;
  *arg_storage = arg;
  arg->d = arg->aligned = (T *)t.data_ptr();
  arg->offset = 0;
  assert(t.dim() == N);
  for (int j = 0; j < N; j++) {
    arg->shape[j] = t.sizes()[j];
    arg->stride[j] = t.stride(j);
  }
  return (void *)arg_storage;
}

at::Tensor LowerAndRun(mlir::ModuleOp module,
                       std::vector<at::Tensor> &arguments, const ir::Value &v,
                       mlir::MLIRContext &context) {

  LowerATenDialect(module);
  LowerStdDialect(module);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel =
      llvm::CodeGenOpt::Level::Aggressive;
  std::string libpath;
  if (const char *path = std::getenv("TEST_BUILD_PATH")) {
    libpath = path;
  }

  std::vector<std::string> sharedLibs{libpath +
                                      "/frontends/pytorch/lib/libaten_ops.so"};
  llvm::errs() << "Loading " << sharedLibs[0] << "\n";

  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

  llvm::SmallVector<llvm::StringRef, 1> libs(sharedLibs.begin(),
                                             sharedLibs.end());
  auto expectedEngine = mlir::ExecutionEngine::create(
      module, {}, jitCodeGenOptLevel, libs, false, false, false);
  assert(expectedEngine && "no engine, cannot fly");

  llvm::StringRef entryPoint("_mlir_ciface_graph");
  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  assert(expectedFPtr && "entryPoint missing");

  void (*fptr)(void **) = *expectedFPtr;

  // this array holds pointers to the function arguments
  void **args = (void **)malloc((arguments.size() + 1) * sizeof(void *));

  // allocate and setup the function arguments
  for (int i = 0, e = arguments.size(); i < e; i++) {
    at::Tensor &t = arguments[i];
    auto dtype = t.dtype();
    int dim = t.dim();
    if (dim == 4) {
      if (dtype == at::kFloat)
        args[i] = setupArg<float, 4>(t);
      else if (dtype == at::kLong)
        args[i] = setupArg<uint64_t, 4>(t);
      else
        assert(0);
    } else if (dim == 3) {
      if (dtype == at::kFloat)
        args[i] = setupArg<float, 3>(t);
      else if (dtype == at::kLong)
        args[i] = setupArg<uint64_t, 3>(t);
      else
        assert(0);
    } else if (dim == 2) {
      if (dtype == at::kFloat)
        args[i] = setupArg<float, 2>(t);
      else if (dtype == at::kLong)
        args[i] = setupArg<uint64_t, 2>(t);
      else
        assert(0);
    } else if (dim == 1) {
      if (dtype == at::kFloat)
        args[i] = setupArg<float, 1>(t);
      else if (dtype == at::kLong)
        args[i] = setupArg<uint64_t, 1>(t);
      else
        assert(0);
    } else {
      assert(0 && "unhandled dim");
    }
  }

  // allocate the result tensors
  // TODO: num results > 1
  at::Tensor result = util::Zeros(v.sizes(), at::kFloat);
  if (result.dim() == 4) {
    args[arguments.size()] = setupArg<float, 4>(result);
  } else if (result.dim() == 3) {
    args[arguments.size()] = setupArg<float, 3>(result);
  } else if (result.dim() == 2) {
    args[arguments.size()] = setupArg<float, 2>(result);
  } else if (result.dim() == 1) {
    args[arguments.size()] = setupArg<float, 1>(result);
  } else {
    assert(0 && "unhandled dim");
  }

  // call the JITed function
  fptr(args);

  // free pointers to the results
  // TODO: num results > 1
  if (result.dim() == 4) {
    auto arg_storage =
        static_cast<llvm_tensor_t<float, 4> **>(args[arguments.size()]);
    auto arg = *arg_storage;
    delete arg;
    delete arg_storage;
  } else if (result.dim() == 3) {
    auto arg_storage =
        static_cast<llvm_tensor_t<float, 3> **>(args[arguments.size()]);
    auto arg = *arg_storage;
    delete arg;
    delete arg_storage;
  } else if (result.dim() == 2) {
    auto arg_storage =
        static_cast<llvm_tensor_t<float, 2> **>(args[arguments.size()]);
    auto arg = *arg_storage;
    delete arg;
    delete arg_storage;
  } else if (result.dim() == 1) {
    auto arg_storage =
        static_cast<llvm_tensor_t<float, 1> **>(args[arguments.size()]);
    auto arg = *arg_storage;
    delete arg;
    delete arg_storage;
  } else {
    assert(0 && "unhandled dim");
  }

  // free pointers to the arguments
  for (int i = 0, e = arguments.size(); i < e; i++) {
    at::Tensor &t = arguments[i];
    int dim = t.dim();
    if (dim == 4) {
      auto arg_storage = static_cast<llvm_tensor_t<float, 4> **>(args[i]);
      auto arg = *arg_storage;
      delete arg;
      delete arg_storage;
    } else if (dim == 3) {
      auto arg_storage = static_cast<llvm_tensor_t<float, 3> **>(args[i]);
      auto arg = *arg_storage;
      delete arg;
      delete arg_storage;
    } else if (dim == 2) {
      auto arg_storage = static_cast<llvm_tensor_t<float, 2> **>(args[i]);
      auto arg = *arg_storage;
      delete arg;
      delete arg_storage;
    } else if (dim == 1) {
      auto arg_storage = static_cast<llvm_tensor_t<float, 1> **>(args[i]);
      auto arg = *arg_storage;
      delete arg;
      delete arg_storage;
    } else {
      assert(0 && "unhandled dim");
    }
  }

  // free the array of void* ptrs
  free(args);

  return result;
}

at::Tensor JitAndRun(const ir::Value &v, mlir::MLIRContext &context) {

  // generate the MLIR
  std::vector<ir::Value> vs{v};
  auto mlir_gen = MLIRGen(context).genModule(vs);
  mlir::OwningModuleRef module = std::move(std::get<0>(mlir_gen));
  std::vector<at::Tensor> arguments = std::move(std::get<1>(mlir_gen));

  return LowerAndRun(module.get(), arguments, v, context);
}

at::Tensor JitAndRun(const ir::Value &v) {
  mlir::MLIRContext context;
  return JitAndRun(v, context);
}

at::Tensor Interpret(const ir::Value &v) { assert(0 && "unsupported"); }
} // anonymous namespace

// FIXME: Why is this code here and not in tensor.cpp?
std::string MLIRTensor::GetMLIR() const {

  // generate the MLIR
  mlir::MLIRContext context;
  ir::Value ir_value = CurrentIrValue();
  if (!ir_value)
    return "<tensor>";

  std::vector<ir::Value> vs{ir_value};
  auto mlir_gen = MLIRGen(context).genModule(vs);
  mlir::OwningModuleRef module = std::move(std::get<0>(mlir_gen));

  std::string aten;
  llvm::raw_string_ostream ss(aten);
  module->print(ss);
  return ss.str();
}

at::Tensor MLIRTensor::CompileAndRun() const {
  return JitAndRun(CurrentIrValue());
}

} // namespace torch_mlir
