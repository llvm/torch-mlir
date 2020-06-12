//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_INITALL_H
#define NPCOMP_INITALL_H

namespace mlir {
namespace NPCOMP {

void registerAllDialects();
void registerAllPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_INITALL_H
