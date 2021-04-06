//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_REFBACKEND_PASSES
#define NPCOMP_REFBACKEND_PASSES

include "mlir/Pass/PassBase.td"

def LowerToRefbackrtABI : Pass<"lower-to-refbackrt-abi", "ModuleOp"> {
  let summary = "Lower constructs requiring runtime support to `refbackrt`";
  let description = [{
    We have a specialized dialect `refbackrt` which models our runtime's data
    structures, and function signatures (and presumably eventually, other
    ABI boundaries like external calls if we ever support it) will be
    converted.

    The constructs requiring runtime support are:
    - function signatures / module metadata
    - error handling
  }];
  let constructor = "mlir::NPCOMP::createLowerToRefbackrtABIPass()";
}

def LowerAllocMemRefOps : Pass<"lower-alloc-memref-ops", "FuncOp"> {
  let summary = "Lower AllocMemRefOp's";
  let constructor = "mlir::NPCOMP::createLowerAllocMemRefOpsPass()";
  let dependentDialects = ["tensor::TensorDialect"];
}

def LowerToLLVM : Pass<"refback-lower-to-llvm", "ModuleOp"> {
  let summary = "Lower everything to LLVM";
  let constructor = "mlir::NPCOMP::createLowerToLLVMPass();";
}

// TODO: Move this pass to upstream.
// TODO: This pass will still do "folding" on all ops.
// The applyPatternsAndFoldGreedily driver will need to be changed to restrict
// folding to the specified dialects as well.
// Perhaps a better design is having a pass that uses the conversion framework.
// The the pass constructor would take a set of op names, and it would
// set up a conversion target that makes all those ops illegal, and uses
// the canonicalization patterns from those ops to legalize them.
def RestrictedCanonicalizer : Pass<"restricted-canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass is the same as the regular `canonicalize` pass, but it only
    applies a restricted set of patterns.

    This is useful when a particular canonicalization is actually needed for
    correctness of a lowering flow. For such cases, running a restricted set of
    canonicalizations makes it clearer which passes are needed for correctness
    and which passes are "just optimizations". This helps when debugging
    miscompiles and other situations where the compiler is not behaving as
    expected.
  }];
  let constructor = "mlir::NPCOMP::createRestrictedCanonicalizerPass()";
  let options = [
    ListOption<"includedDialects", "included-dialects", "std::string",
               "Which dialects should be canonicalized",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def RecognizeTorchFallback : Pass<"refback-recognize-torch-fallback", "FuncOp"> {
  let summary = "Encapsulates torch.kernel_call ops.";
  let constructor = "mlir::NPCOMP::createRecognizeTorchFallbackPass()";
}

#endif // NPCOMP_REFBACKEND_PASSES
