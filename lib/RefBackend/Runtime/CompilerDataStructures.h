//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains data structures (which we typically call "descriptors")
// that are emitted by the compiler and must be kept in sync with the compiler
// code that creates them in LowerToLLVM.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_LIB_RUNTIME_COMPILERDATASTRUCTURES_H
#define NPCOMP_LIB_RUNTIME_COMPILERDATASTRUCTURES_H

#include <cstdint>

namespace refbackrt {

// All arguments are packed into this type-erased form for being invoked. See
// LowerToLLVM.cpp for more details.
typedef void ABIFunc(void **, void **);

enum class ABIArgType : std::uint32_t {
  kNone = 0,
  kMemref,
  kF32,
  kF64,
};

enum class ABIElementType : std::uint32_t {
  kNone = 0,
  kF32,
};

struct InputDescriptor {
  ABIArgType abiType;
  ABIElementType elementType;

  std::int32_t rank;
  std::int32_t* extents;

  // TODO(brycearden): Change to bool at ABI boundary
  // std::int32_t isStatic;
};

struct OutputDescriptor {
  ABIArgType abiType;
  ABIElementType elementType;

  std::int32_t rank;
  std::int32_t* extents;

  // TODO(brycearden): Change to bool at ABI boundary
  //std::int32_t isStatic;
};

struct FuncDescriptor {
  // The length of the function name.
  std::int32_t nameLen;
  // The name of the function, to allow lookup.
  const char *name;
  // This is a raw function pointer to the function's entry point as
  // emitted by the compiler.
  ABIFunc *functionPtr;
  // The number of inputs to the function.
  std::int32_t numInputs;
  // The number of outputs of the function.
  std::int32_t numOutputs;
  // TODO: Add shape checking to arg / result descriptor(s)
  InputDescriptor *inputDescriptors;
  OutputDescriptor *outputDescriptors;
};

// The top-level entry point of the module metadata emitted by the
// compiler. Unlike all the other descriptors here, external code does handle
// this type (albeit through an opaque pointer).
struct ModuleDescriptor {
  std::int32_t numFuncDescriptors;
  FuncDescriptor *functionDescriptors;
};

} // namespace refbackrt

#endif // NPCOMP_LIB_RUNTIME_COMPILERDATASTRUCTURES_H
