//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/Passes.h"

#include "npcomp/Conversion/TorchToIREE/TorchToIREE.h"
#include "npcomp/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "npcomp/Conversion/TorchToSCF/TorchToSCF.h"
#include "npcomp/Conversion/TorchToStd/TorchToStd.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Conversion/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::registerConversionPasses() { ::registerPasses(); }
