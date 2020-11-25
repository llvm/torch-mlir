//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using mlir::LLVM::LLVMFuncOp;
using mlir::LLVM::LLVMType;

//===----------------------------------------------------------------------===//
// Descriptor types shared with the runtime.
//
// These correspond to the types in CompilerDataStructures.h
//===----------------------------------------------------------------------===//

// Get the LLVMType for refbackrt::FuncDescriptor.
static LLVMType getFuncDescriptorTy(MLIRContext *context) {
  return LLVMType::getStructTy(context, {
                                            // Name length.
                                            LLVMType::getIntNTy(context, 32),
                                            // Name chars.
                                            LLVMType::getInt8PtrTy(context),
                                            // Type-erased function pointer.
                                            LLVMType::getInt8PtrTy(context),
                                            // Number of inputs.
                                            LLVMType::getIntNTy(context, 32),
                                            // Number of outputs.
                                            LLVMType::getIntNTy(context, 32),
                                        });
}

// Get the LLVMType for refbackrt::ModuleDescriptor.
static LLVMType getModuleDescriptorTy(MLIRContext *context) {
  return LLVMType::getStructTy(context,
                               {
                                   // std::int32_t numFuncDescriptors;
                                   LLVMType::getIntNTy(context, 32),
                                   // FuncDescriptor *functionDescriptors;
                                   getFuncDescriptorTy(context).getPointerTo(),
                               });
}

//===----------------------------------------------------------------------===//
// Compiler runtime functions.
//===----------------------------------------------------------------------===//

namespace {
template <typename T>
class TrivialCompilerRuntimeLowering : public OpConversionPattern<T> {
public:
  TrivialCompilerRuntimeLowering(LLVM::LLVMFuncOp backingFunc)
      : OpConversionPattern<T>(backingFunc.getContext()),
        backingFunc(backingFunc) {}
  LogicalResult
  matchAndRewrite(T op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, backingFunc, operands);
    return success();
  }
  LLVM::LLVMFuncOp backingFunc;
};
} // namespace

static LLVM::GlobalOp createGlobalString(ModuleOp module, StringAttr msg,
                                         OpBuilder &builder, Location loc) {
  // TODO: Deduplicate strings.
  std::string msgNulTerminated = msg.getValue().str();
  msgNulTerminated.push_back('\0');
  auto arrayTy = LLVMType::getArrayTy(LLVMType::getInt8Ty(module.getContext()),
                                      msgNulTerminated.size());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  // To get a unique symbol name, use a suffix derived from the current number
  // of ops in the module.
  // We can't use the SymbolTable's logic for this because the module
  // transiently contains a `func` and `llvm.func` with the same name during
  // conversion, preventing us from instantiating a SymbolTable.
  std::string symbolName =
      (Twine("__npcomp_string_") +
       Twine(llvm::size(llvm::to_vector<6>(module.getOps<LLVM::GlobalOp>()))))
          .str();
  auto globalOp = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal, symbolName,
      builder.getStringAttr(msgNulTerminated));
  return globalOp;
}

namespace {
class AbortIfOpCompilerRuntimeLowering
    : public OpConversionPattern<refbackrt::AbortIfOp> {
public:
  AbortIfOpCompilerRuntimeLowering(LLVM::LLVMFuncOp backingFunc)
      : OpConversionPattern<refbackrt::AbortIfOp>(backingFunc.getContext()),
        backingFunc(backingFunc) {}
  LogicalResult
  matchAndRewrite(refbackrt::AbortIfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    refbackrt::AbortIfOp::Adaptor adaptor(operands);
    auto *context = op.getContext();

    // Create the global string, take its address, and gep to get an `i8*`.
    auto globalOp = createGlobalString(op.getParentOfType<ModuleOp>(),
                                       op.msgAttr(), rewriter, op.getLoc());
    auto msgArray = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalOp);
    auto c0 = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), LLVMType::getIntNTy(context, 32),
        rewriter.getI32IntegerAttr(0));
    auto msg = rewriter.create<LLVM::GEPOp>(op.getLoc(),
                                            LLVMType::getInt8PtrTy(context),
                                            msgArray, ValueRange({c0, c0}));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, backingFunc, ValueRange({adaptor.pred(), msg}));
    return success();
  }
  LLVM::LLVMFuncOp backingFunc;
};
} // namespace

// Create the LLVM runtime function backing the refbackrt op with name `name`
// and requiring `type`.
static LLVMFuncOp createCompilerRuntimeFuncDecl(StringRef name, LLVMType type,
                                                OpBuilder &builder,
                                                Location loc) {
  assert(type.isFunctionTy());
  std::string symbolName = (Twine("__npcomp_compiler_rt_") + name).str();
  return builder.create<LLVM::LLVMFuncOp>(loc, symbolName, type,
                                          LLVM::Linkage::External);
}

static void populateCompilerRuntimePatterns(ModuleOp module,
                                            OwningRewritePatternList &patterns,
                                            LLVMTypeConverter &typeConverter) {
  auto *context = module.getContext();
  OpBuilder builder(module.getBodyRegion());

  {
    auto abortIfFuncTy = LLVMType::getFunctionTy(
        LLVMType::getVoidTy(context),
        {LLVMType::getInt1Ty(context), LLVMType::getInt8PtrTy(context)},
        /*isVarArg=*/false);
    LLVMFuncOp abortIfFunc = createCompilerRuntimeFuncDecl(
        "abort_if", abortIfFuncTy, builder, module.getLoc());
    patterns.insert<AbortIfOpCompilerRuntimeLowering>(abortIfFunc);
  }
}

//===----------------------------------------------------------------------===//
// Lowering for module metadata
//===----------------------------------------------------------------------===//

static LLVM::GlobalOp
createFuncDescriptorArray(ArrayRef<refbackrt::FuncMetadataOp> funcMetadatas,
                          OpBuilder &builder, Location loc) {
  auto llvmI32Ty = LLVMType::getIntNTy(builder.getContext(), 32);

  DenseMap<StringRef, LLVM::GlobalOp> globalsByName;
  for (auto funcMetadata : funcMetadatas) {
    auto arrayTy =
        LLVMType::getArrayTy(LLVMType::getInt8Ty(builder.getContext()),
                             funcMetadata.funcName().size());
    std::string llvmSymbolName =
        (Twine("__npcomp_internal_constant_") + funcMetadata.funcName()).str();
    auto global = builder.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        llvmSymbolName, builder.getStringAttr(funcMetadata.funcName()));
    globalsByName[funcMetadata.funcName()] = global;
  }

  // This must match FuncDescriptor in the runtime.
  auto funcDescriptorTy = getFuncDescriptorTy(builder.getContext());
  auto funcDescriptorArrayTy =
      LLVMType::getArrayTy(funcDescriptorTy, funcMetadatas.size());
  auto funcDescriptorArrayGlobal = builder.create<LLVM::GlobalOp>(
      loc, funcDescriptorArrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
      "__npcomp_func_descriptors",
      /*value=*/Attribute());
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&funcDescriptorArrayGlobal.initializer());

  // Build the initializer.
  Value funcDescriptorArray =
      builder.create<LLVM::UndefOp>(loc, funcDescriptorArrayTy);
  auto updateDescriptor = [&](Value value,
                              std::initializer_list<int32_t> position) {
    funcDescriptorArray = builder.create<LLVM::InsertValueOp>(
        loc, funcDescriptorArray, value,
        /*position=*/builder.getI32ArrayAttr(position));
  };
  auto updateDescriptorWithI32Attr =
      [&](Attribute attr, std::initializer_list<int32_t> position) {
        auto constant = builder.create<LLVM::ConstantOp>(loc, llvmI32Ty, attr);
        updateDescriptor(constant, position);
      };
  auto c0 = builder.create<LLVM::ConstantOp>(loc, llvmI32Ty,
                                             builder.getI32IntegerAttr(0));
  for (auto funcMetadataAndIndex : llvm::enumerate(funcMetadatas)) {
    auto funcMetadata = funcMetadataAndIndex.value();
    int32_t index = funcMetadataAndIndex.index();

    // Name length.
    updateDescriptorWithI32Attr(
        builder.getI32IntegerAttr(funcMetadata.funcName().size()), {index, 0});

    // Name chars.
    auto funcNameArray = builder.create<LLVM::AddressOfOp>(
        loc, globalsByName[funcMetadata.funcName()]);
    auto funcNamePtr = builder.create<LLVM::GEPOp>(
        loc, LLVMType::getInt8PtrTy(builder.getContext()), funcNameArray,
        ValueRange({c0, c0}));
    updateDescriptor(funcNamePtr, {index, 1});

    // Function pointer.
    //
    // We create this reference to the original function (and use a dummy i8*
    // type). We will fix this up after conversion to point at wrapper
    // functions that satisfy the ABI requirements.
    // The bitcast is required so that after conversion the inserted value is an
    // i8* as expected by the descriptor struct.
    auto funcAddress = builder.create<LLVM::AddressOfOp>(
        loc, LLVMType::getInt8PtrTy(builder.getContext()),
        funcMetadata.funcName());
    auto typeErasedFuncAddress = builder.create<LLVM::BitcastOp>(
        loc, LLVMType::getInt8PtrTy(builder.getContext()), funcAddress);
    updateDescriptor(typeErasedFuncAddress, {index, 2});

    // Number of inputs.
    updateDescriptorWithI32Attr(funcMetadata.numInputsAttr(), {index, 3});

    // Number of outputs.
    updateDescriptorWithI32Attr(funcMetadata.numOutputsAttr(), {index, 4});
  }

  builder.create<LLVM::ReturnOp>(loc, funcDescriptorArray);

  return funcDescriptorArrayGlobal;
}

LLVM::GlobalOp createModuleDescriptor(LLVM::GlobalOp funcDescriptorArray,
                                      OpBuilder &builder, Location loc) {
  auto llvmI32Ty = LLVMType::getIntNTy(builder.getContext(), 32);
  auto moduleDescriptorTy = getModuleDescriptorTy(builder.getContext());
  // TODO: Ideally this symbol name would somehow be related to the module
  // name, if we could consistently assume we had one.
  // TODO: We prepend _mlir so that mlir::ExecutionEngine's lookup logic (which
  // is typically only mean for function pointers) will find this raw symbol.
  auto moduleDescriptorGlobal = builder.create<LLVM::GlobalOp>(
      loc, moduleDescriptorTy, /*isConstant=*/true, LLVM::Linkage::External,
      "_mlir___npcomp_module_descriptor",
      /*value=*/Attribute());
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&moduleDescriptorGlobal.initializer());

  Value moduleDescriptor =
      builder.create<LLVM::UndefOp>(loc, moduleDescriptorTy);

  auto updateDescriptor = [&](Value value,
                              std::initializer_list<int32_t> position) {
    moduleDescriptor = builder.create<LLVM::InsertValueOp>(
        loc, moduleDescriptor, value,
        /*position=*/builder.getI32ArrayAttr(position));
  };

  updateDescriptor(
      builder.create<LLVM::ConstantOp>(
          loc, llvmI32Ty,
          builder.getI32IntegerAttr(
              funcDescriptorArray.getType().getArrayNumElements())),
      {0});

  auto funcDecriptorArrayAddress =
      builder.create<LLVM::AddressOfOp>(loc, funcDescriptorArray);
  auto rawFuncDescriptorPtr = builder.create<LLVM::BitcastOp>(
      loc, getFuncDescriptorTy(builder.getContext()).getPointerTo(),
      funcDecriptorArrayAddress);
  updateDescriptor(rawFuncDescriptorPtr, {1});
  builder.create<LLVM::ReturnOp>(loc, moduleDescriptor);

  return moduleDescriptorGlobal;
}

namespace {
class LowerModuleMetadata
    : public OpConversionPattern<refbackrt::ModuleMetadataOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(refbackrt::ModuleMetadataOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcMetadatas =
        llvm::to_vector<6>(op.metadatas().getOps<refbackrt::FuncMetadataOp>());
    auto funcDescriptorArray =
        createFuncDescriptorArray(funcMetadatas, rewriter, op.getLoc());
    auto moduleDescriptor =
        createModuleDescriptor(funcDescriptorArray, rewriter, op.getLoc());

    // TODO: create get module descriptor wrapper (or upgrade
    // mlir::ExecutionEngine to allow raw symbol lookup)
    (void)moduleDescriptor;

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// Performs the calculation:
// ```
// ty *f(void **voidStarStar, int32_t i) {
//   return reinterpret_cast<ty *>(voidStarStar[i]);
// }
// ```
static Value getTypedAddressFromVoidStarStar(Value voidStarStar, int32_t index,
                                             LLVMType ty, OpBuilder &builder,
                                             Location loc) {
  Value ci = builder.create<LLVM::ConstantOp>(
      loc, LLVMType::getIntNTy(builder.getContext(), 32),
      builder.getI32IntegerAttr(index));

  // Do `voidStarStar[i]` as a gep + load.
  auto inputPtrAddr = builder.create<LLVM::GEPOp>(
      loc, LLVMType::getInt8PtrTy(builder.getContext()).getPointerTo(),
      voidStarStar, ValueRange(ci));
  auto inputPtr = builder.create<LLVM::LoadOp>(loc, inputPtrAddr);
  return builder.create<LLVM::BitcastOp>(loc, ty.getPointerTo(), inputPtr);
}

static SmallVector<Value, 6> loadCallArgs(Value inputsPtrPtr, LLVMType funcTy,
                                          OpBuilder &builder, Location loc) {
  SmallVector<Value, 6> callArgs;
  // For each void* in the void**, cast it to the right type and load it.
  for (int i = 0, e = funcTy.getFunctionNumParams(); i < e; i++) {
    auto paramTy = funcTy.getFunctionParamType(i);
    auto addr =
        getTypedAddressFromVoidStarStar(inputsPtrPtr, i, paramTy, builder, loc);
    callArgs.push_back(builder.create<LLVM::LoadOp>(loc, addr));
  }
  return callArgs;
}

static LLVM::LLVMType getUnrankedMemrefDescriptorType(MLIRContext *context) {
  LLVMTypeConverter converter(context);
  // LLVMTypeConverter doesn't directly expose the struct type used to represent
  // unranked memrefs on ABI boundaries. To get that type, we convert
  // an unranked memref type and see what it produces.
  //
  // An unranked memref is just a size_t for the rank and an void* pointer to
  // descriptor, so the choice of element type here is arbitrary -- it all
  // converts to the same thing.
  return converter
      .convertType(UnrankedMemRefType::get(Float32Type::get(context),
                                           /*memorySpace=*/0))
      .cast<LLVM::LLVMType>();
}

// Writes out the logical results of the wrapper function through the void**
// passed on the ABI boundary. Because LLVM (and hence llvm.func)
// only supports a single return type (or void/no results), the logic here needs
// to be aware of the convention used in the Std to LLVM conversion to map
// multiple return types. The details of this are in the function
// packFunctionResults and its callers:
// https://github.com/llvm/llvm-project/blob/fad9cba8f58ba9979f390a49cf174ec9fcec29a6/mlir/lib/Conversion/StandardToLLVM/StandardToLLVM.cpp#L282
static void storeWrapperResults(LLVM::CallOp callToWrapped, Value resultsPtrPtr,
                                OpBuilder &builder, Location loc) {
  // 0 results. Nothing to do.
  if (callToWrapped.getNumResults() == 0)
    return;
  Value result = callToWrapped.getResult(0);
  auto ty = result.getType().cast<LLVMType>();
  // 1 logical result.
  if (ty == getUnrankedMemrefDescriptorType(ty.getContext())) {
    Value addr =
        getTypedAddressFromVoidStarStar(resultsPtrPtr, 0, ty, builder, loc);
    builder.create<LLVM::StoreOp>(loc, result, addr);
    return;
  }
  assert(ty.isStructTy() && "must be a multi-result packed struct!");
  // >=2 logical results. The convention linked above will create a struct
  // wrapping.
  for (int i = 0, e = ty.getStructNumElements(); i < e; i++) {
    auto elementTy = ty.getStructElementType(i);
    Value addr = getTypedAddressFromVoidStarStar(resultsPtrPtr, i, elementTy,
                                                 builder, loc);
    int32_t i32I = i;
    Value value = builder.create<LLVM::ExtractValueOp>(
        loc, elementTy, result, builder.getI32ArrayAttr({i32I}));
    builder.create<LLVM::StoreOp>(loc, value, addr);
  }
}

// Construct a wrapper function.
// For an externally visible function f(T1, T2) -> T3, T4, we create a
// wrapper
// __refbackrt_wrapper_f(void **inputs, void ** outputs) {
//  T3 t3;
//  T4 t4;
//  (t3, t4) = f(*cast<T1*>(inputs[0]), *cast<T2*>(inputs[1]));
//  *cast<T3*>(outputs[0]) = t3;
//  *cast<T4*>(outputs[1]) = t4;
// }
// This is very similar to MLIR's "packed" convention, but supporting
// outputs.
// TODO: Extend MLIR's void** wrappers to have outputs in this way.
static LLVMFuncOp createWrapperFunc(LLVMFuncOp func) {
  auto *context = func.getContext();
  LLVMType funcTy = func.getType();
  auto voidStarTy = LLVMType::getInt8PtrTy(context);
  auto voidStarStarTy = voidStarTy.getPointerTo();
  auto wrapperTy = LLVMType::getFunctionTy(LLVMType::getVoidTy(context),
                                           {voidStarStarTy, voidStarStarTy},
                                           /*isVarArg=*/false);
  constexpr char kRefbackrtWrapperPrefix[] = "__refbackrt_wrapper_";
  auto wrapperName = (Twine(kRefbackrtWrapperPrefix) + func.getName()).str();
  OpBuilder moduleBuilder(func.getParentRegion());
  LLVMFuncOp wrapper = moduleBuilder.create<LLVMFuncOp>(
      func.getLoc(), wrapperName, wrapperTy, LLVM::Linkage::External);

  // Create the function body.
  Block &body = *wrapper.addEntryBlock();
  auto builder = OpBuilder::atBlockBegin(&body);
  auto callArgs =
      loadCallArgs(body.getArgument(0), funcTy, builder, func.getLoc());
  auto call = builder.create<LLVM::CallOp>(func.getLoc(), func, callArgs);
  storeWrapperResults(call, body.getArgument(1), builder, func.getLoc());
  builder.create<LLVM::ReturnOp>(func.getLoc(), ValueRange());
  return wrapper;
}

namespace {
class LowerToLLVM : public LowerToLLVMBase<LowerToLLVM> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    LLVMTypeConverter converter(context);

    OwningRewritePatternList patterns;
    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    populateCompilerRuntimePatterns(module, patterns, converter);
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<LowerModuleMetadata>(context);

    // TODO: Move these "std to std" legalizations to their own pass if we grow
    // lots of these patterns.
    populateExpandTanhPattern(patterns, context);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    // Rewrite llvm.mlir.addressof ops that reference the original exported
    // functions from the module to instead refer to wrapper functions.
    // These wrapper functions have a fixed ABI
    // (`void f(void **inputs, void **results)`) which we can interface to from
    // external code without dealing with platform-dependent
    // register-level calling conventions. We embed enough information in the
    // module metadata to make sure that calling code can e.g. preallocate
    // enough outputs and with the right types to safely funnel through this
    // convention.
    module.walk([&](LLVM::AddressOfOp op) {
      auto originalFunc =
          module.lookupSymbol<LLVM::LLVMFuncOp>(op.global_name());
      if (!originalFunc)
        return;
      auto wrapper = createWrapperFunc(originalFunc);
      op.getResult().setType(wrapper.getType().getPointerTo());
      Builder builder(op.getContext());
      op.setAttr("global_name", builder.getSymbolRefAttr(wrapper.getName()));
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::NPCOMP::createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVM>();
}
