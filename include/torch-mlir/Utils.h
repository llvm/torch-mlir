//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_UTILS_H
#define TORCHMLIR_UTILS_H

class Endian {
private:
  static constexpr uint32_t uint32_ = 0x01020304;
  static constexpr uint8_t magic_ = (const uint8_t &)uint32_;

public:
  static constexpr bool little = magic_ == 0x04;
  static constexpr bool big = magic_ == 0x01;
  static_assert(little || big, "Cannot determine endianness!");

private:
  Endian() = delete;
};

#endif // TORCHMLIR_UTILS_H
