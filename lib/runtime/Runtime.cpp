//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/runtime/UserAPI.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "CompilerDataStructures.h"

using namespace npcomprt;

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

std::int32_t npcomprt::getElementTypeByteSize(ElementType type) {
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

void npcomprt::invoke(ModuleDescriptor *moduleDescriptor,
                      StringRef functionName, ArrayRef<Ref<Tensor>> inputs,
                      MutableArrayRef<Ref<Tensor>> outputs) {
  auto *descriptor = getFuncDescriptor(moduleDescriptor, functionName);
  assert(descriptor && "unknown function name");
  assert(inputs.size() < kMaxArity && "number of inputs exceeds kMaxArity");
  assert(outputs.size() < kMaxArity && "number of outputs exceeds kMaxArity");
  std::array<Tensor *, kMaxArity> inputTensorPtrs;
  std::array<Tensor *, kMaxArity> outputTensorPtrs;
  std::array<void *, kMaxArity> packedInputs;
  std::array<void *, kMaxArity> packedOutputs;
  for (int i = 0, e = inputs.size(); i < e; i++)
    inputTensorPtrs[i] = inputs[i].get();
  for (int i = 0, e = inputs.size(); i < e; i++)
    packedInputs[i] = ToVoidPtr(inputTensorPtrs[i]);
  descriptor->functionPtr(packedInputs.data(), packedOutputs.data());
  for (int i = 0, e = outputs.size(); i < e; i++)
    outputTensorPtrs[i] = static_cast<Tensor *>(packedOutputs[i]);
  // TODO: Actually manage refcounts inside the compiler.
  // Right now, we only pass around npcomprt.tensor's in trivial ways on ABI
  // boundaries, so the following contract of the compiler-generated code works:
  // - input tensors are never retained or released
  // - output tensors always have refcount 0. Hence the next line here is
  // actually essential because it increments the refcounts so they are nonzero.
  for (int i = 0, e = outputs.size(); i < e; i++)
    outputs[i] = Ref<Tensor>(outputTensorPtrs[i]);
}

LogicalResult npcomprt::getMetadata(ModuleDescriptor *moduleDescriptor,
                                    StringRef functionName,
                                    FunctionMetadata &outMetadata) {
  auto *descriptor = getFuncDescriptor(moduleDescriptor, functionName);
  if (!descriptor)
    return failure();
  outMetadata.numInputs = descriptor->numInputs;
  outMetadata.numOutputs = descriptor->numOutputs;
  return success();
}