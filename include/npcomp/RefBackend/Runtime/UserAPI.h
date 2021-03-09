//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the public-facing interface for interacting with the npcomp
// runtime.
//
// This functionality is totally firewalled from the compiler codebase, so
// even if things superficially look similar, remember that there are no
// LLVM utilities here, memory allocation should be kept to a minimum, etc.
//
// npcomp/RefBackend/Runtime/Support.h provides some minimal LLVM-like support
// code to keep the API familiar.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_RUNTIME_USERAPI_H
#define NPCOMP_RUNTIME_USERAPI_H

#include "npcomp/RefBackend/Runtime/Support.h"
#include <atomic>
#include <cstdlib>
#include <string>

namespace refbackrt {

struct RtValue;

// Base class for any RefCounted object type
class RefTarget {
  mutable std::atomic<size_t> refcount;

  template <typename T> friend class Ref;
  friend struct RtValue;
  inline void incref() const { this->refcount += 1; }

  template <typename T> friend class Ref;
  friend struct RtValue;
  inline bool decref() const {
    if (this->refcount.fetch_sub(1) == 1)
      return true;
    return false;
  }

public:
  size_t refCount() const { return refcount; }

protected:
  void setRefCount(uint32_t val) { refcount = val; }

  constexpr RefTarget() noexcept : refcount(0) {}
};

// Reference-counted handle to a type with a `refCount` member.
// T is expected to be a RefTarget
template <typename T> class Ref {
public:
  Ref() { ptr = nullptr; }
  // Creates a Ref and increments the refcount by 1.
  // rawPtr must be allocated with std::malloc.
  Ref(T *rawPtr) {
    assert(rawPtr->refCount() >= 0 &&
           "expected non-negative refcount to start!");
    ptr = rawPtr;
    ptr->incref();
  }
  Ref(const Ref &other) {
    ptr = other.ptr;
    ptr->incref();
  }
  Ref(Ref &&other) { ptr = other.takePtr(); }
  Ref &operator=(const Ref &other) {
    if (&other == this)
      return *this;
    if (ptr != nullptr && ptr->decref())
      releaseResources();
    ptr = other.ptr;
    ptr->incref();
    return *this;
  }
  Ref &operator=(Ref &&other) {
    if (&other == this)
      return *this;
    if (ptr != nullptr && ptr->decref()) {
      releaseResources();
    }
    ptr = other.takePtr();
    return *this;
  }
  ~Ref() {
    if (ptr != nullptr && ptr->decref()) {
      releaseResources();
    }
  }

  T &operator*() const { return *ptr; }
  T *operator->() const { return ptr; }
  T *get() const { return ptr; }

  T *takePtr() {
    auto *ret = ptr;
    ptr = nullptr;
    return ret;
  }

  static Ref reclaimPtr(T *otherPtr) {
    assert(otherPtr->refCount() >= 1);
    auto ret = Ref();
    ret.ptr = otherPtr;
    return ret;
  }

  int debugGetRefCount() { return ptr->refCount(); }

private:
  void releaseResources() {
    if (ptr == nullptr) {
      assert(false && "ptr is nullptr");
    }
    ptr->~T();
    std::free(ptr);
  }

  T *ptr;
};

// The available data types.
enum class ElementType : std::int32_t {
  F32,
};
std::int32_t getElementTypeByteSize(ElementType type);

// Representation of a tensor.
class Tensor : public RefTarget {
public:
  // Due to tail-allocated objects, this struct should never be directly
  // constructed.
  Tensor() = delete;

  // Create a Tensor with the given extents and element type, with a buffer
  // holding a copy of `data`.
  static Ref<Tensor> create(ArrayRef<std::int32_t> extents,
                            ElementType elementType, void *data);
  // Same as `create`, but returns a raw pointer.
  static Tensor *createRaw(ArrayRef<std::int32_t> extents,
                           ElementType elementType, void *data);

  ElementType getElementType() const { return elementType; }
  std::int32_t getRank() const { return rank; }
  void *getData() const { return data; }
  template <typename T> T *getData() const { return static_cast<T *>(data); }
  std::int32_t getExtent(int dimension) const {
    return getExtents()[dimension];
  }
  ArrayRef<std::int32_t> getExtents() const {
    auto extents = const_cast<Tensor *>(this)->getMutableExtents();
    return ArrayRef<std::int32_t>(extents.data(), extents.size());
  }
  // Returns the number of bytes occupied by the data representing this tensor.
  // The total allocated amount might be higher to allow e.g. for alignment
  // nudging.
  std::int32_t getDataByteSize() const;
  ~Tensor() { std::free(allocatedPtr); }

private:
  MutableArrayRef<std::int32_t> getMutableExtents() {
    auto *tail = reinterpret_cast<std::int32_t *>(this + 1);
    return MutableArrayRef<std::int32_t>(tail, rank);
  }
  // Reference count management.
  // template <typename T> friend class Ref;
  // friend struct RtValue;
  // std::atomic<int> refCount{0};

  ElementType elementType;
  // The number of dimensions of this Tensor.
  // There are `rank` tail-allocated std::int32_t values representing the
  // tensor extents.
  std::int32_t rank;
  // The buffer base.
  void *data;
  // The raw pointer returned by the allocator (currently assumed to be
  // malloc), suitable for freeing the buffer.
  void *allocatedPtr;

  // Sizes are tail-allocated.
};

// RtValue is a generic tagged union used to hold all value types
// The tag determines the type, and the payload represents the stored
// contents of an object. If an object is not trivially destructible,
// then it must be refcounted and must have a refCount.
#define NPCOMP_FORALL_TAGS(_)                                                  \
  _(None)                                                                      \
  _(Bool)                                                                      \
  _(Int)                                                                       \
  _(Double)                                                                    \
  _(Tensor)

struct RtValue final {

  RtValue() : payload{0}, tag(Tag::None) {}

  // Bool
  RtValue(bool b) : tag(Tag::Bool) { payload.asBool = b; }
  bool isBool() const { return Tag::Bool == tag; }
  bool toBool() const {
    assert(isBool());
    return payload.asBool;
  }

  // Int
  RtValue(std::int64_t i) : tag(Tag::Int) { payload.asInt = i; }
  RtValue(std::int32_t i) : RtValue(static_cast<int64_t>(i)) {}
  bool isInt() const { return Tag::Int == tag; }
  bool toInt() const {
    assert(isInt());
    return payload.asInt;
  }

  // Double
  RtValue(double d) : tag(Tag::Double) { payload.asDouble = d; }
  bool isDouble() const { return Tag::Double == tag; }
  bool toDouble() const {
    assert(isDouble());
    return payload.asDouble;
  }

  // Tensor
  RtValue(Ref<Tensor> tensor) : tag(Tag::Tensor) {
    payload.asRefTargetPtr = reinterpret_cast<RefTarget *>(tensor.takePtr());
  }
  bool isTensor() const { return Tag::Tensor == tag; }
  Ref<Tensor> toTensor() const {
    assert(isTensor());
    return Ref<Tensor>(reinterpret_cast<Tensor *>(payload.asRefTargetPtr));
  }

  // Ref
  bool isRef() const { return isTensor(); }

  // RtValue (downcast)
  const RtValue &toRtValue() const { return *this; }
  RtValue &toRtValue() { return *this; }

  // Stringify tag for debugging.
  std::string tagKind() const {
    switch (tag) {
#define DEFINE_CASE(x)                                                         \
  case Tag::x:                                                                 \
    return #x;
      NPCOMP_FORALL_TAGS(DEFINE_CASE)
#undef DEFINE_CASE
    }
    // TODO(brycearden): Print tag here
    return "InvalidTag!";
  }

  RtValue(const RtValue &rhs) : RtValue(rhs.payload, rhs.tag) {
    if (isRef()) {
      payload.asRefTargetPtr->incref();
    }
  }
  RtValue(RtValue &&rhs) noexcept : RtValue() { swap(rhs); }

  RtValue &operator=(RtValue &&rhs) & noexcept {
    RtValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  RtValue &operator=(RtValue const &rhs) & {
    RtValue(rhs).swap(*this);
    return *this;
  }

  ~RtValue() {
    if (isRef()) {
      if (isTensor()) {
        auto raii = Ref<Tensor>::reclaimPtr(
            reinterpret_cast<Tensor *>(payload.asRefTargetPtr));
        return;
      }
      assert(false && "Unsupported RtValue type");
    }
  }

private:
  void swap(RtValue &rhs) {
    std::swap(payload, rhs.payload);
    std::swap(tag, rhs.tag);
  }

  // NOTE: Runtime tags are intentionally private.
  // Please use the helper functions above to query information about the type
  // of a RtValue.
  enum class Tag : std::uint32_t {
#define DEFINE_TAG(x) x,
    NPCOMP_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
  };

  union Payload {
    bool asBool;
    int64_t asInt;
    double asDouble;
    RefTarget *asRefTargetPtr;
  };

  RtValue(Payload pl, Tag tag) : payload(pl), tag(tag) {}

  Payload payload;
  Tag tag;
};

//===----------------------------------------------------------------------===//
// Module loading.
// This is the main entry point that users interact with.
//===----------------------------------------------------------------------===//

// Metadata for a particular function.
// TODO: Add arg types.
struct FunctionMetadata {
  std::int32_t numInputs;
  std::int32_t numOutputs;
};

// Opaque forward declaration of module descriptor type. This is the type
// created by the compiler in the module binary.
struct ModuleDescriptor;

// Maximum input or output arity.
constexpr static int kMaxArity = 20;

// Low-level invocation API. The number of inputs and outputs should be correct
// and match the results of getMetadata.
void invoke(ModuleDescriptor *moduleDescriptor, StringRef functionName,
            ArrayRef<RtValue> inputs, MutableArrayRef<RtValue> outputs);

// Metadata for function `functionName`.
//
// Returns failure if functionName wasn't found.
LogicalResult getMetadata(ModuleDescriptor *moduleDescriptor,
                          StringRef functionName,
                          FunctionMetadata &outMetadata);

} // namespace refbackrt

#endif // NPCOMP_RUNTIME_USERAPI_H
