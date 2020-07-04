//===- Interfaces.h - Interfaces for IR types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_CPA_INTERFACES_H
#define NPCOMP_TYPING_CPA_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "npcomp/Typing/CPA/Support.h"

namespace mlir {

#include "npcomp/Typing/CPA/OpInterfaces.h.inc"
#include "npcomp/Typing/CPA/TypeInterfaces.h.inc"

} // namespace mlir

#endif // NPCOMP_TYPING_CPA_INTERFACES_H
