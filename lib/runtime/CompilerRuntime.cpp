//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbols referenced only by the compiler and which will be compiled into a
// shared object that a JIT can load to provide those symbols.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdlib>
#include <iostream>

#include "CompilerDataStructures.h"
#include "npcomp/runtime/UserAPI.h"

using namespace npcomprt;

extern "C" void __npcomp_compiler_rt_abort_if(bool b) {
  if (b) {
    std::fprintf(stderr, "NPCOMP: aborting!\n");
    std::exit(1);
  }
}

extern "C" std::size_t __npcomp_compiler_rt_get_extent(Tensor *tensor,
                                                       std::int32_t dim) {
  assert(dim < tensor->getRank() && "dim out of bounds!");
  return tensor->getExtent(dim);
}

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
};

struct UnrankedMemref {
  int64_t rank;
  MemrefDescriptor *descriptor;
};

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

std::int32_t getNumElements(MemrefDescriptor *descriptor, int assumedRank) {
  if (assumedRank == 0)
    return 1;
  return descriptor->getSizes(assumedRank)[0] *
         descriptor->getStrides(assumedRank)[0];
}

} // namespace

extern "C" UnrankedMemref __npcomp_compiler_rt_to_memref(Tensor *tensor) {
  auto byteSize = tensor->getDataByteSize();
  void *data = std::malloc(byteSize);
  std::memcpy(data, tensor->getData(), byteSize);
  auto *descriptor = MemrefDescriptor::create(tensor->getExtents(), data);
  return UnrankedMemref{tensor->getRank(), descriptor};
}

extern "C" Tensor *
__npcomp_compiler_rt_from_memref(std::int64_t rank,
                                 MemrefDescriptor *descriptor) {
  auto numElements = getNumElements(descriptor, rank);
  // TODO: Have the compiler pass this as an argument.
  auto elementType = ElementType::F32;

  auto byteSize = getElementTypeByteSize(elementType) * numElements;
  void *data = std::malloc(byteSize);
  std::memcpy(data, descriptor->dataPtr, byteSize);
  auto extents64 = descriptor->getSizes(rank);

  // Launder from std::int64_t to std::int32_t.
  constexpr int kMaxRank = 20;
  std::array<std::int32_t, kMaxRank> extents32Buf;
  for (int i = 0, e = extents64.size(); i < e; i++)
    extents32Buf[i] = extents64[i];
  return Tensor::createRaw(ArrayRef<std::int32_t>(extents32Buf.data(), rank),
                           elementType, data);
}

extern "C" UnrankedMemref
__npcomp_compiler_rt_get_global(GlobalDescriptor *global) {
  auto *descriptor = MemrefDescriptor::create(
      ArrayRef<std::int32_t>(global->extents, global->numExtents),
      global->data);
  return UnrankedMemref{global->numExtents, descriptor};
}