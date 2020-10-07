//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Trimmed down support classes intended to provide a familiar LLVM-like API,
// but without actually pulling in the LLVM ones.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_RUNTIME_SUPPORT_H
#define NPCOMP_RUNTIME_SUPPORT_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace npcomprt {
class StringRef {
public:
  StringRef(const char *ptr, std::size_t length) : ptr(ptr), length(length){};
  // Construct from NUL-terminated C string.
  StringRef(const char *ptr) : ptr(ptr), length(std::strlen(ptr)) {}
  bool equals(StringRef other) {
    if (length != other.length)
      return false;
    return std::memcmp(ptr, other.ptr, length) == 0;
  }

private:
  const char *ptr;
  std::size_t length;
};
inline bool operator==(StringRef lhs, StringRef rhs) { return lhs.equals(rhs); }
inline bool operator!=(StringRef lhs, StringRef rhs) {
  return !operator==(lhs, rhs);
}

template <typename T> class ArrayRef {
public:
  ArrayRef(const T *ptr, std::size_t length) : ptr(ptr), length(length){};
  const T &operator[](std::size_t i) const {
    assert(i < length);
    return ptr[i];
  }
  const T *data() const { return ptr; }
  std::size_t size() const { return length; }

private:
  const T *ptr;
  std::size_t length;
};

template <typename T> class MutableArrayRef {
public:
  MutableArrayRef(T *ptr, std::size_t length) : ptr(ptr), length(length){};
  T &operator[](std::size_t i) {
    assert(i < length);
    return ptr[i];
  }
  T *data() const { return ptr; }
  std::size_t size() const { return length; }

private:
  T *ptr;
  std::size_t length;
};

// Literally copied from MLIR.
struct LogicalResult {
  enum ResultEnum { Success, Failure } value;
  LogicalResult(ResultEnum v) : value(v) {}
};

inline LogicalResult success(bool isSuccess = true) {
  return LogicalResult{isSuccess ? LogicalResult::Success
                                 : LogicalResult::Failure};
}

inline LogicalResult failure(bool isFailure = true) {
  return LogicalResult{isFailure ? LogicalResult::Failure
                                 : LogicalResult::Success};
}

inline bool succeeded(LogicalResult result) {
  return result.value == LogicalResult::Success;
}

inline bool failed(LogicalResult result) {
  return result.value == LogicalResult::Failure;
}

} // namespace npcomprt

#endif // NPCOMP_RUNTIME_SUPPORT_H
