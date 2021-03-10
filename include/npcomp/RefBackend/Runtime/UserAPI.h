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
#include <array>
#include <atomic>
#include <cstdlib>

namespace refbackrt {

struct RtValue;

// Base class for any RefCounted object type
class RefTarget {
protected:
  template <typename T> friend class Ref;
  mutable std::atomic<size_t> refCount;

  constexpr RefTarget() noexcept : refCount(0) {}
};

// Reference-counted handle to a type with a `refCount` member.
template <typename T> class Ref {
public:
  Ref() { ptr = nullptr; }
  // Creates a Ref and increments the refcount by 1.
  // rawPtr must be allocated with std::malloc.
  Ref(T *rawPtr) {
    assert(rawPtr->refCount >= 0 && "expected non-negative refcount to start!");
    ptr = rawPtr;
    incref(ptr);
  }
  Ref(const Ref &other) {
    ptr = other.ptr;
    incref(ptr);
  }
  Ref(Ref &&other) { ptr = other.takePtr(); }
  Ref &operator=(const Ref &other) {
    if (&other == this)
      return *this;
    decref(ptr);
    ptr = other.ptr;
    incref(ptr);
    return *this;
  }
  Ref &operator=(Ref &&other) {
    if (&other == this)
      return *this;
    decref(ptr);
    ptr = other.takePtr();
    return *this;
  }
  ~Ref() { decref(ptr); }

  T &operator*() const { return *ptr; }
  T *operator->() const { return ptr; }
  T *get() const { return ptr; }

  T *takePtr() {
    auto *ret = ptr;
    ptr = nullptr;
    return ret;
  }

  int debugGetRefCount() { return ptr->refCount; }

private:
  friend struct RtValue;
  static void incref(T *ptr) {
    if (!ptr)
      return;
    ptr->refCount += 1;
  }

  friend struct RtValue;
  static void decref(T *ptr) {
    if (!ptr)
      return;
    if (ptr->refCount.fetch_sub(1) == 1) {
      ptr->~T();
      std::free(static_cast<void *>(ptr));
    }
  }
  T *ptr;
};

// The available data types.
enum class ElementType : std::int32_t {
  NONE,
  F32,
};
std::int32_t getElementTypeByteSize(ElementType type);
StringRef getElementTypeAsStringRef(ElementType type);

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

  static Ref<Tensor> create(ArrayRef<std::int64_t> extents,
                            ElementType elementType, void *data);
  // Same as `create`, but returns a raw pointer.
  static Tensor *createRaw(ArrayRef<std::int64_t> extents,
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
#define NPCOMP_FORALL_PRIM_TAGS(_)                                             \
  _(None)                                                                      \
  _(Bool)                                                                      \
  _(Int)                                                                       \
  _(Float)                                                                     \
  _(Double)

#define NPCOMP_FORALL_REF_TAGS(_) _(Tensor)

#define NPCOMP_FORALL_TAGS(_)                                                  \
  NPCOMP_FORALL_PRIM_TAGS(_)                                                   \
  NPCOMP_FORALL_REF_TAGS(_)

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
  int64_t toInt() const {
    assert(isInt());
    return payload.asInt;
  }

  // Float
  RtValue(float f) : tag(Tag::Float) { payload.asFloat = f; }
  bool isFloat() const { return Tag::Float == tag; }
  float toFloat() const {
    assert(isFloat());
    return payload.asFloat;
  }

  // Double
  RtValue(double d) : tag(Tag::Double) { payload.asDouble = d; }
  bool isDouble() const { return Tag::Double == tag; }
  double toDouble() const {
    assert(isDouble());
    return payload.asDouble;
  }

  // Tensor
  RtValue(Ref<Tensor> tensor) : tag(Tag::Tensor) {
    payload.asVoidPtr = reinterpret_cast<void *>(tensor.takePtr());
  }
  bool isTensor() const { return Tag::Tensor == tag; }
  Ref<Tensor> toTensor() const {
    assert(isTensor());
    return Ref<Tensor>(reinterpret_cast<Tensor *>(payload.asVoidPtr));
  }

  // Ref
  bool isRef() const {
#define DEFINE_IS_REF(x)                                                       \
  if (is##x()) {                                                               \
    return true;                                                               \
  }
    NPCOMP_FORALL_REF_TAGS(DEFINE_IS_REF)
#undef DEFINE_IS_REF
    return false;
  }

  // Scalar
  bool isScalar() const {
    return isBool() || isInt() || isFloat() || isDouble();
  }

  // RtValue (downcast)
  const RtValue &toRtValue() const { return *this; }
  RtValue &toRtValue() { return *this; }

  // Stringify tag for debugging.
  StringRef tagKind() const {
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
#define DEFINE_INCREF(x)                                                       \
  if (is##x()) {                                                               \
    Ref<x>::incref(static_cast<x *>(payload.asVoidPtr));                       \
    return;                                                                    \
  }
      NPCOMP_FORALL_REF_TAGS(DEFINE_INCREF)
#undef DEFINE_INCREF
      assert(false && "Unsupported RtValue type");
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
#define DEFINE_DECREF(x)                                                       \
  if (is##x()) {                                                               \
    Ref<x>::decref(static_cast<x *>(payload.asVoidPtr));                       \
    return;                                                                    \
  }
      NPCOMP_FORALL_REF_TAGS(DEFINE_DECREF)
#undef DEFINE_DECREF
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
    float asFloat;
    double asDouble;
    void *asVoidPtr;
  };

  RtValue(Payload pl, Tag tag) : payload(pl), tag(tag) {}

  Payload payload;
  Tag tag;
};

//===----------------------------------------------------------------------===//
// Module loading.
// This is the main entry point that users interact with.
//===----------------------------------------------------------------------===//

enum class ArgType : std::uint32_t {
  kNone = 0,
  kTensor,
  kF32,
  kF64,
};
StringRef getArgTypeAsStringRef(ArgType type);

// Maximum rank supported across the ABI boundary
constexpr static int kMaxRank = 6;

struct InputArgInfo {
  // What type of argument this is
  ArgType argType;
  // Certain arg types also have an element type
  ElementType elementType;
  std::int32_t rank;
  std::array<std::int32_t, kMaxRank> extents;
};

struct OutputArgInfo {
  // What type of argument this is
  ArgType argType;
  // Certain arg types also have an element type
  ElementType elementType;
  std::int32_t rank;
  std::array<std::int32_t, kMaxRank> extents;
  // TODO(brycearden): Add checks for whether output buffers alias to input
  // buffers and populate field(s) here indicating that case
};

// Maximum input or output arity.
constexpr static int kMaxArity = 20;

// Metadata for a particular function.
struct FunctionMetadata {
  std::int32_t numInputs;
  std::int32_t numOutputs;

  std::array<InputArgInfo, kMaxArity> inputArgInfos;
  std::array<OutputArgInfo, kMaxArity> outputArgInfos;
};

// Opaque forward declaration of module descriptor type. This is the type
// created by the compiler in the module binary.
struct ModuleDescriptor;

// Verifies that the input RtValue arg types match what the user provides
// matches the types we expect from the descriptors emitted by the
// compiler.
//
// Returns failure if the input type(s) are not valid
LogicalResult checkRtValueArgTypes(const RtValue &value,
                                   const InputArgInfo &info);

// Verifies that the input RtValue shapes matches what the user provides
// matches the types we expect from the descriptors emitted by the
// compiler.
//
// Returns failure if the input type(s) are not valid
LogicalResult checkRtValueShapes(const RtValue &value,
                                 const InputArgInfo &info);

// Creates an RtValue of the right type from the output metadata
// provided by the compiled module
RtValue createRtValueFromOutputArgInfo(const OutputArgInfo &info);

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
