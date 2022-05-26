//===- import_options.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_H
#define TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_H

namespace torch_mlir {
// Common import options across importers. We define this as a struct to avoid
// an unstructured proliferation of different kinds of ways to control different
// parts of the import process.
struct ImportOptions {
  // If this is set to true, then all tensors in the program can be assumed to
  // have value semantics. This can happen, for example, when coming from
  // LazyTensorCore since conversion to value semantics has already happened at
  // a higher level there before we see the program. For
  // calling-convention-impacting decisions, this flag should be interpreted as
  // a requirement to use a value-semantic tensor type (!torch.vtensor) in
  // signatures.
  bool assumeTensorsHaveValueSemantics = false;
};
} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_H
