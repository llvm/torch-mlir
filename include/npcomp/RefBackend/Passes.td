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

def BypassShapes : Pass<"bypass-shapes", "FuncOp"> {
  let summary = "Bypass shape calculations around ops";
  let constructor = "mlir::NPCOMP::createBypassShapesPass()";
}

def LowerShapedResultsToMemref : Pass<"lower-shaped-results-to-memref", "FuncOp"> {
  let summary = "Lower refback.shaped_results regions";
  let constructor = "mlir::NPCOMP::createLowerShapedResultsToMemrefPass()";
}

def LowerStdToMemref : Pass<"lower-std-to-memref", "FuncOp"> {
  let summary = "Lower std ops to memref";
  let constructor = "mlir::NPCOMP::createLowerStdToMemrefPass()";
}

def LowerConstantTensorsToMemref :
    Pass<"lower-constant-tensors-to-memref", "ModuleOp"> {
  let summary = "Lower std.constant of tensor type to memref";
  let description = [{
    This must be a module pass since it involves creating refback.global ops.
  }];
  let constructor = "mlir::NPCOMP::createLowerConstantTensorsToMemrefPass()";
}

def LowerStructuralToMemref :
    Pass<"lower-structural-to-memref", "FuncOp"> {
  let summary = "Lower structural IR constructs to memref";
  let description = [{
    Structural constructs include:
      - control flow ops (both CFG and SCF)
      - function signatures
      - TODO: calls
    An op is "structural" if it doesn't really care about the types it operates
    on, but the types just have to converted to be consistent.

    This pass also cleans up any previous memref<->tensor materializations,
    finalizing the conversion from tensor to memref.
  }];
  let constructor = "mlir::NPCOMP::createLowerStructuralToMemrefPass()";
}

def LowerToNpcomprtABI : Pass<"lower-to-npcomprt-abi", "ModuleOp"> {
  let summary = "Lower constructs requiring runtime support to `npcomprt`";
  let description = [{
    We have a specialized dialect `npcomprt` which models our runtime's data
    structures, and function signatures (and presumably eventually, other
    ABI boundaries like external calls if we ever support it) will be
    converted.

    The constructs requiring runtime support are:
    - function signatures / module metadata
    - globals
    - error handling
  }];
  let constructor = "mlir::NPCOMP::createLowerToNpcomprtABIPass()";
}

def LowerAllocMemRefOps : Pass<"lower-alloc-memref-ops", "FuncOp"> {
  let summary = "Lower AllocMemRefOp's";
  let constructor = "mlir::NPCOMP::createLowerAllocMemRefOpsPass()";
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

#endif // NPCOMP_REFBACKEND_PASSES
