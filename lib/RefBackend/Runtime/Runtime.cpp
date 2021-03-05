//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/RefBackend/Runtime/UserAPI.h"

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
  case ElementType::F32:
    return 4;
  }
}

Ref<Tensor> Tensor::create(ArrayRef<std::int32_t> extents, ElementType type,
                           void *data) {
  return Ref<Tensor>(createRaw(extents, type, data));
}

Tensor *Tensor::createRaw(ArrayRef<std::int32_t> extents, ElementType type,
                          void *data) {
  auto *tensor = static_cast<Tensor *>(
      std::malloc(sizeof(Tensor) + extents.size() * sizeof(std::int32_t)));

  tensor->refCount.store(0);
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
                       StringRef functionName, ArrayRef<RuntimeValue> inputs,
                       MutableArrayRef<RuntimeValue> outputs) {
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
  for (int i = 0, e = inputs.size(); i < e; i++) {
    inputUnrankedMemrefs[i] =
        convertRefbackrtTensorToUnrankedMemref(inputs[i].toTensor().get());
  }
  // Create a type-erased list of "packed inputs" to pass to the
  // LLVM/C ABI wrapper function. Each packedInput pointer corresponds to
  // one LLVM/C ABI argument to the underlying function.
  //
  // The ABI lowering on StandardToLLVM conversion side will
  // "explode" the unranked memref descriptors on the underlying function
  // into separate arguments for the rank and pointer-to-descriptor.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    packedInputs[2 * i] = ToVoidPtr(&inputUnrankedMemrefs[i].rank);
    packedInputs[2 * i + 1] = ToVoidPtr(&inputUnrankedMemrefs[i].descriptor);
  }
  // Create a type-erased list of "packed output" to pass to the
  // LLVM/C ABI wrapper function.
  //
  // Due to how StandardToLLVM lowering works, each packedOutput pointer
  // corresponds to a single UnrankedMemref (not "exploded").
  for (int i = 0, e = outputs.size(); i < e; i++) {
    packedOutputs[i] = ToVoidPtr(&outputUnrankedMemrefs[i]);
  }

  // Actually invoke the function!
  descriptor->functionPtr(packedInputs.data(), packedOutputs.data());

  // Copy out the result data into refbackrt::Tensor's.
  // TODO: Avoid needing to make a deep copy.
  for (int i = 0, e = outputs.size(); i < e; i++) {
    // TODO: Have compiler emit the element type in the metadata.
    auto elementType = ElementType::F32;
    Tensor *tensor = convertUnrankedMemrefToRefbackrtTensor(
        outputUnrankedMemrefs[i].rank, outputUnrankedMemrefs[i].descriptor,
        elementType);
    outputs[i] = RuntimeValue(Ref<Tensor>(tensor));
  }

  // Now, we just need to free all the UnrankedMemref's that we created.
  // This is complicated by the fact that multiple input/output UnrankedMemref's
  // can end up with the same backing buffer (`allocatedPtr`), and we need
  // to avoid double-freeing.
  // Output buffers might alias any other input or output buffer.
  // Input buffers are guaranteed to not alias each other.

  // Free the output buffers.
  for (int i = 0, e = outputs.size(); i < e; i++) {
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

  // Free the input buffers.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    void *allocatedPtr = inputUnrankedMemrefs[i].descriptor->allocatedPtr;
    bool bufferNeedsFreeing = true;
    for (int j = 0, je = outputs.size(); j < je; j++) {
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
    // The LLVM lowering guarantees that each returned unranked memref
    // descriptor is separately malloc'ed, so no need to do anything special
    // like we had to do for the allocatedPtr's.
    std::free(outputUnrankedMemrefs[i].descriptor);
  }
  // Free the input descriptors.
  for (int i = 0, e = inputs.size(); i < e; i++) {
    std::free(inputUnrankedMemrefs[i].descriptor);
  }
}

LogicalResult refbackrt::getMetadata(ModuleDescriptor *moduleDescriptor,
                                     StringRef functionName,
                                     FunctionMetadata &outMetadata) {
  auto *descriptor = getFuncDescriptor(moduleDescriptor, functionName);
  if (!descriptor)
    return failure();
  outMetadata.numInputs = descriptor->numInputs;
  outMetadata.numOutputs = descriptor->numOutputs;
  return success();
}
