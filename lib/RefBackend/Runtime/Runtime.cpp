//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/RefBackend/Runtime/UserAPI.h"

#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "CompilerDataStructures.h"

using namespace refbackrt;

//===----------------------------------------------------------------------===//
// Memref descriptors for interacting with MLIR codegenerated code.
//===----------------------------------------------------------------------===//

namespace {
// These definitions are based on the ones in
// `mlir/ExecutionEngine/CRunnerUtils.h` and the layouts need to be kept in
// sync.
//
// Those definitions are flawed though because they are overly templated.
struct MemrefDescriptor {
  void *allocatedPtr;
  void *dataPtr;
  std::int64_t offset;
  // Tail-allocated int64_t sizes followed by strides.
  MutableArrayRef<std::int64_t> getSizes(int assumedRank) {
    auto *tail = reinterpret_cast<std::int64_t *>(this + 1);
    return MutableArrayRef<std::int64_t>(tail, assumedRank);
  }
  MutableArrayRef<std::int64_t> getStrides(int assumedRank) {
    auto *tail = reinterpret_cast<std::int64_t *>(this + 1);
    return MutableArrayRef<std::int64_t>(tail + assumedRank, assumedRank);
  }

  // Returns a malloc-allocated MemrefDescriptor with the specified extents and
  // default striding.
  static MemrefDescriptor *create(ArrayRef<std::int32_t> extents, void *data);

  // Returns the number of elements in this MemrefDescriptor, assuming this
  // descriptor has rank `assumedRank`.
  std::int32_t getNumElements(int assumedRank) {
    if (assumedRank == 0)
      return 1;
    return getSizes(assumedRank)[0] * getStrides(assumedRank)[0];
  }
};
} // namespace

namespace {
struct UnrankedMemref {
  int64_t rank;
  MemrefDescriptor *descriptor;
};
} // namespace

MemrefDescriptor *MemrefDescriptor::create(ArrayRef<std::int32_t> extents,
                                           void *data) {
  auto rank = extents.size();
  auto allocSize = sizeof(MemrefDescriptor) + sizeof(std::int64_t) * 2 * rank;
  auto *descriptor = static_cast<MemrefDescriptor *>(std::malloc(allocSize));
  descriptor->allocatedPtr = data;
  descriptor->dataPtr = data;
  descriptor->offset = 0;
  // Iterate in reverse, copying the dimension sizes (i.e. extents) and
  // calculating the strides for a standard dense layout.
  std::int64_t stride = 1;
  for (int i = 0, e = rank; i < e; i++) {
    auto revIdx = e - i - 1;
    descriptor->getSizes(rank)[revIdx] = extents[revIdx];
    descriptor->getStrides(rank)[revIdx] = stride;
    stride *= extents[revIdx];
  }
  return descriptor;
}

static UnrankedMemref convertRefbackrtTensorToUnrankedMemref(Tensor *tensor) {
  auto byteSize = tensor->getDataByteSize();
  void *data = std::malloc(byteSize);
  std::memcpy(data, tensor->getData(), byteSize);
  auto *descriptor = MemrefDescriptor::create(tensor->getExtents(), data);
  return UnrankedMemref{tensor->getRank(), descriptor};
}

static Tensor *convertUnrankedMemrefToRefbackrtTensor(
    std::int64_t rank, MemrefDescriptor *descriptor, ElementType elementType) {
  // Launder from std::int64_t to std::int32_t.
  auto extents64 = descriptor->getSizes(rank);
  constexpr int kMaxRank = 20;
  std::array<std::int32_t, kMaxRank> extents32Buf;
  for (int i = 0, e = extents64.size(); i < e; i++)
    extents32Buf[i] = extents64[i];
  return Tensor::createRaw(ArrayRef<std::int32_t>(extents32Buf.data(), rank),
                           elementType, descriptor->dataPtr);
}

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

static std::int32_t totalElements(ArrayRef<std::int32_t> extents) {
  std::int32_t ret = 1;
  for (int i = 0, e = extents.size(); i < e; i++) {
    ret *= extents[i];
  }
  return ret;
}

std::int32_t refbackrt::getElementTypeByteSize(ElementType type) {
  switch (type) {
  case ElementType::NONE:
    return 0;
  case ElementType::F32:
    return 4;
  }
  llvm_unreachable("unsupported dtype");
}

StringRef refbackrt::getElementTypeAsStringRef(ElementType type) {
  switch (type) {
  case ElementType::NONE:
    return "NONE";
  case ElementType::F32:
    return "F32";
  }
  llvm_unreachable("unsupported element type string");
}

StringRef refbackrt::getArgTypeAsStringRef(ArgType type) {
  switch (type) {
  case ArgType::kNone:
    return "kNone";
  case ArgType::kTensor:
    return "kTensor";
  case ArgType::kF32:
    return "kF32";
  case ArgType::kF64:
    return "kF64";
  }
  llvm_unreachable("unsupported arg type string");
}

Ref<Tensor> Tensor::create(ArrayRef<std::int32_t> extents, ElementType type,
                           void *data) {
  return Ref<Tensor>(createRaw(extents, type, data));
}

Tensor *Tensor::createRaw(ArrayRef<std::int32_t> extents, ElementType type,
                          void *data) {
  auto *tensor = static_cast<Tensor *>(
      std::malloc(sizeof(Tensor) + extents.size() * sizeof(std::int32_t)));

  tensor->refCount = 0;
  tensor->elementType = type;
  tensor->rank = extents.size();
  auto byteSize = getElementTypeByteSize(type) * totalElements(extents);
  // TODO: Align the buffer.
  tensor->allocatedPtr = std::malloc(byteSize);
  tensor->data = tensor->allocatedPtr;
  std::memcpy(tensor->data, data, byteSize);
  for (int i = 0, e = extents.size(); i < e; i++)
    tensor->getMutableExtents()[i] = extents[i];
  return tensor;
}

std::int32_t Tensor::getDataByteSize() const {
  return getElementTypeByteSize(getElementType()) * totalElements(getExtents());
}

//===----------------------------------------------------------------------===//
// Module metadata descriptors.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Module operations.
//===----------------------------------------------------------------------===//

template <typename T> static void *ToVoidPtr(T *ptr) {
  return const_cast<void *>(static_cast<const void *>(ptr));
}
static FuncDescriptor *getFuncDescriptor(ModuleDescriptor *moduleDescriptor,
                                         StringRef name) {
  for (int i = 0, e = moduleDescriptor->numFuncDescriptors; i < e; i++) {
    auto &functionDescriptor = moduleDescriptor->functionDescriptors[i];
    if (StringRef(functionDescriptor.name, functionDescriptor.nameLen) ==
        name) {
      return &functionDescriptor;
    }
  }
  return nullptr;
}

void refbackrt::invoke(ModuleDescriptor *moduleDescriptor,
                       StringRef functionName, ArrayRef<RtValue> inputs,
                       MutableArrayRef<RtValue> outputs) {
  auto *descriptor = getFuncDescriptor(moduleDescriptor, functionName);
  assert(descriptor && "unknown function name");
  assert(inputs.size() < kMaxArity && "number of inputs exceeds kMaxArity");
  assert(outputs.size() < kMaxArity && "number of outputs exceeds kMaxArity");

  // We haven't committed to using "vector" in this runtime code, so use
  // a fixed-sized array.
  std::array<UnrankedMemref, kMaxArity> inputUnrankedMemrefs;
  std::array<UnrankedMemref, kMaxArity> outputUnrankedMemrefs;
  std::array<void *, kMaxArity * 2> packedInputs;
  std::array<void *, kMaxArity> packedOutputs;

  // Deepcopy the refbackrt::Tensor's into UnrankedMemref's.
  // TODO: Avoid the deep copy. It makes the later lifetime management code
  // more complex though (and maybe impossible given the current abstractions).
  //
  // Create a type-erased list of "packed inputs" to pass to the
  // LLVM/C ABI wrapper function. Each packedInput pointer corresponds to
  // one LLVM/C ABI argument to the underlying function.
  //
  // The ABI lowering on StandardToLLVM conversion side will
  // "explode" the unranked memref descriptors on the underlying function
  // into separate arguments for the rank and pointer-to-descriptor.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    auto idx = 2 * i;
    if (inputs[i].isTensor()) {
      inputUnrankedMemrefs[i] =
          convertRefbackrtTensorToUnrankedMemref(inputs[i].toTensor().get());
      packedInputs[idx] = ToVoidPtr(&inputUnrankedMemrefs[i].rank);
      packedInputs[idx + 1] = ToVoidPtr(&inputUnrankedMemrefs[i].descriptor);
    } else if (inputs[i].isScalar()) {
      packedInputs[idx] = ToVoidPtr(&inputs[i]);
    } else {
      assert(false && "unsupported input RtValue type");
    }
  }

  // Create a type-erased list of "packed output" to pass to the
  // LLVM/C ABI wrapper function.
  //
  // Due to how StandardToLLVM lowering works, each packedOutput pointer
  // corresponds to a single UnrankedMemref (not "exploded").
  for (int i = 0, e = outputs.size(); i < e; i++) {
    if (outputs[i].isTensor()) {
      packedOutputs[i] = ToVoidPtr(&outputUnrankedMemrefs[i]);
    } else if (outputs[i].isScalar()) {
      packedOutputs[i] = ToVoidPtr(&outputs[i]);
    }
  }

  // Actually invoke the function!
  descriptor->functionPtr(packedInputs.data(), packedOutputs.data());

  // Copy out the result data into refbackrt::Tensor's.
  // TODO: Avoid needing to make a deep copy.
  for (int i = 0, e = outputs.size(); i < e; i++) {
    // TODO: Have compiler emit the element type in the metadata.
    if (outputs[i].isTensor()) {
      auto elementType = ElementType::F32;
      Tensor *tensor = convertUnrankedMemrefToRefbackrtTensor(
          outputUnrankedMemrefs[i].rank, outputUnrankedMemrefs[i].descriptor,
          elementType);
      outputs[i] = RtValue(Ref<Tensor>(tensor));
    } else if (outputs[i].isFloat()) {
      outputs[i] = RtValue(*(reinterpret_cast<float *>(packedOutputs[i])));
    }
  }

  // Now, we just need to free all the UnrankedMemref's that we created.
  // This is complicated by the fact that multiple input/output UnrankedMemref's
  // can end up with the same backing buffer (`allocatedPtr`), and we need
  // to avoid double-freeing.
  // Output buffers might alias any other input or output buffer.
  // Input buffers are guaranteed to not alias each other.

  // Free the output buffers.
  for (int i = 0, e = outputs.size(); i < e; i++) {
    if (outputs[i].isRef()) {
      void *allocatedPtr = outputUnrankedMemrefs[i].descriptor->allocatedPtr;
      // Multiple returned memrefs can point into the same underlying
      // malloc allocation. Do a linear scan to see if any of the previously
      // deallocated buffers already freed this pointer.
      bool bufferNeedsFreeing = true;
      for (int j = 0; j < i; j++) {
        if (allocatedPtr == outputUnrankedMemrefs[j].descriptor->allocatedPtr)
          bufferNeedsFreeing = false;
      }
      if (!bufferNeedsFreeing)
        std::free(allocatedPtr);
    }
  }

  // Free the input buffers.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    if (!inputs[i].isRef())
      continue;
    void *allocatedPtr = inputUnrankedMemrefs[i].descriptor->allocatedPtr;
    bool bufferNeedsFreeing = true;
    for (int j = 0, je = outputs.size(); j < je; j++) {
      if (!outputs[j].isRef())
        continue;
      if (allocatedPtr == outputUnrankedMemrefs[j].descriptor->allocatedPtr)
        bufferNeedsFreeing = false;
    }
    // HACK: The returned memref can point into statically allocated memory that
    // we can't pass to `free`, such as the result of lowering a tensor-valued
    // `std.constant` to `std.global_memref`. The LLVM lowering of
    // std.global_memref sets the allocated pointer to the magic value
    // 0xDEADBEEF, which we sniff for here. This is yet another strong signal
    // that memref is really not the right abstraction for ABI's.
    if (reinterpret_cast<std::intptr_t>(allocatedPtr) == 0xDEADBEEF)
      bufferNeedsFreeing = false;
    if (!bufferNeedsFreeing)
      std::free(allocatedPtr);
  }

  // Free the output descriptors.
  for (int i = 0, e = outputs.size(); i < e; i++) {
    if (!outputs[i].isRef())
      continue;
    // The LLVM lowering guarantees that each returned unranked memref
    // descriptor is separately malloc'ed, so no need to do anything special
    // like we had to do for the allocatedPtr's.
    std::free(outputUnrankedMemrefs[i].descriptor);
  }
  // Free the input descriptors.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    if (!inputs[i].isRef())
      continue;
    std::free(inputUnrankedMemrefs[i].descriptor);
  }
}

static InputArgInfo
getExternalInputArgInfo(const refbackrt::InputDescriptor &inputDescriptor) {
  InputArgInfo ret;

  // Set arg / element types accordingly
  switch (inputDescriptor.abiType) {
  case ABIArgType::kNone:
    ret.argType = ArgType::kNone;
    ret.elementType = ElementType::NONE;
    break;
  case ABIArgType::kMemref:
    ret.argType = ArgType::kTensor;
    ret.elementType = ElementType::F32;
    break;
  case ABIArgType::kF32:
    ret.argType = ArgType::kF32;
    ret.elementType = ElementType::NONE;
    break;
  case ABIArgType::kF64:
    ret.argType = ArgType::kF64;
    ret.elementType = ElementType::NONE;
    break;
  }

  // Extract shape information
  ret.rank = inputDescriptor.rank;
  for (int i = 0; i < inputDescriptor.rank; i++) {
    ret.extents[i] = inputDescriptor.extents[i];
  }

  return ret;
}

static OutputArgInfo
getExternalOutputArgInfo(const refbackrt::OutputDescriptor &outputDescriptor) {
  OutputArgInfo ret;

  // Set arg / element types accordingly
  switch (outputDescriptor.abiType) {
  case ABIArgType::kNone:
    ret.argType = ArgType::kNone;
    ret.elementType = ElementType::NONE;
    break;
  case ABIArgType::kMemref:
    ret.argType = ArgType::kTensor;
    ret.elementType = ElementType::F32;
    break;
  case ABIArgType::kF32:
    ret.argType = ArgType::kF32;
    ret.elementType = ElementType::NONE;
    break;
  case ABIArgType::kF64:
    ret.argType = ArgType::kF64;
    ret.elementType = ElementType::NONE;
    break;
  }

  // Extract shape information
  ret.rank = outputDescriptor.rank;
  for (int i = 0; i < outputDescriptor.rank; i++) {
    ret.extents[i] = outputDescriptor.extents[i];
  }
  return ret;
}

LogicalResult refbackrt::getMetadata(ModuleDescriptor *moduleDescriptor,
                                     StringRef functionName,
                                     FunctionMetadata &outMetadata) {
  auto *descriptor = getFuncDescriptor(moduleDescriptor, functionName);
  if (!descriptor)
    return failure();
  outMetadata.numInputs = descriptor->numInputs;
  outMetadata.numOutputs = descriptor->numOutputs;

  for (int i = 0; i < descriptor->numInputs; i++) {
    outMetadata.inputArgInfos[i] =
        getExternalInputArgInfo(descriptor->inputDescriptors[i]);
  }

  for (int i = 0; i < descriptor->numOutputs; i++) {
    outMetadata.outputArgInfos[i] =
        getExternalOutputArgInfo(descriptor->outputDescriptors[i]);
  }

  return success();
}

LogicalResult refbackrt::checkRtValueShapes(const RtValue &value,
                                            const InputArgInfo &info) {
  if (value.isTensor()) {
    auto refTensor = value.toTensor();

    // Don't bother checking shapes for unranked tensors
    if (info.rank < 0)
      return success();

    if (refTensor->getRank() != info.rank)
      return failure();

    auto tensorExtents = refTensor->getExtents();
    for (int i = 0; i < info.rank; i++) {
      // If a dimension is dynamic, it is encoded as extent = -1
      // and we should skip checking over that dimension
      if (info.extents[i] > 0 && (info.extents[i] != tensorExtents[i]))
        return failure();
    }
  }

  return success();
}

LogicalResult refbackrt::checkRtValueArgTypes(const RtValue &value,
                                              const InputArgInfo &info) {
  // Generic checks based on argType(s)
  if ((value.isTensor() && info.argType != ArgType::kTensor) ||
      (value.isFloat() && info.argType != ArgType::kF32))
    return failure();

  if (value.isRef()) {
    // Will need special error checking for ref-counted types
    // Currently only f32 tensors are supported
    if (value.isTensor()) {
      auto refTensor = value.toTensor();
      if (refTensor->getElementType() != ElementType::F32)
        return failure();
    } else {
      assert(false && "Unsupported input type checking for Ref type");
    }
  }
  return success();
}

RtValue refbackrt::createRtValueFromOutputArgInfo(const OutputArgInfo &info) {
  constexpr int32_t kDynamicConstantShape = 100;
  switch (info.argType) {
  case ArgType::kTensor: {
    // HACK: for dynamic dims the shape will be negative, so for now we are
    // just going to create a tensor of size kDynamicConstantShape
    std::array<int32_t, kMaxRank> tensorShape;
    for (int i = 0; i < info.rank; i++) {
      tensorShape[i] =
          info.extents[i] > 0 ? info.extents[i] : kDynamicConstantShape;
    }
    refbackrt::ArrayRef<int32_t> shape(tensorShape.data(), info.rank);
    int numel = 1;
    for (int i = 0; i < info.rank; i++)
      numel *= shape[i];

    void *data;
    switch (info.elementType) {
    case ElementType::F32: {
      auto byteSize = numel * sizeof(float);
      data = static_cast<void *>(aligned_alloc(32, byteSize));
      memset(data, 0, byteSize);
      return RtValue(Tensor::create(shape, ElementType::F32, data));
      break;
    }
    default: {
      assert(false && "unknown output tensor type");
      return RtValue();
    }
    }

    // The Tensor::create function will malloc and memcpy the data
    // into the Tensor object, so after we need to free our
    // temporary data buffer
    assert(data && "data ptr must exist");
    auto refTensor = Tensor::create(shape, ElementType::F32, data);
    free(data);
    return RtValue(refTensor);
  }
  case ArgType::kF32: {
    return RtValue(-20.0f);
  }
  default: {
    assert(false && "Don't know how to handle this artType");
    return RtValue();
  }
  }
}
