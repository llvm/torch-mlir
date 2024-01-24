//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_TORCH_PASSES
#define TORCHMLIR_TORCH_PASSES

include "mlir/Pass/PassBase.td"

def GlobalizeObjectGraph : Pass<"torch-globalize-object-graph", "ModuleOp"> {
  let summary = "Converts TorchScript object graphs to a globalized form";
  let constructor = "mlir::torch::Torch::createGlobalizeObjectGraphPass()";
  let description = [{
    This pass converts a subset of possible TorchScript modules into a
    more restrictive lower-level form that strips away the need to be
    concerned with instances of !torch.nn.Module<...> type. Specifically,
    the object graph is flattened into a set of discrete globals
    (`torch.global_slot`) that hold the program state.

    The overarching goal is for a strict correspondence between the original
    `torch.nn.Module` (call it `root`) that the user `torch.jit.script`'ed, and
    the public interface of the resulting MLIR module. Specifically:
      - The call `root.encoder.forward(...)` in Python corresponds to invoking
        the `func @encoder.forward` on the resulting MLIR module.
      - The data member access `root.decoder.ids_to_strings_table` in Python
        corresponds to accessing the
        `torch.global_slot @decoder.ids_to_strings_table` on the resulting
        MLIR module.
    In effect, the entire MLIR module corresponds to an instance of the `root`
    object. This matches with the intuitive behavior desired for deployment:
    When the MLIR module (or, more likely, a compiled artifact derived from it)
    is loaded in a deployed environment, it is equivalent to recreating the
    original `root` object.

    This pass performs a complete change of the externally visible calling
    convention of the MLIR module for a graph of objects and methods to a
    fixed set of globals and functions. Additionally, method signatures are
    changed such that all types of !torch.nn.Module are deleted from public
    interfaces since they are guaranteed to correspond to a unique instance and
    are thus redundant.

    Of course, only a subset of programs can be transformed, and this pass fails
    with an error if the conditions are violated.

    Specifically, the restrictions are:
    - There must be a unique torch.nn_module that is not the value of a slot
      of any other torch.nn_module
      - Rationale: Allows us to have a notion of a unique "root" op, which is
        used to define linkage. This also matches how TorchScript imports in
        practice (`torch.jit.script` imports a single root object).
    - Multiple instances of the same class type are allowed, as long as it is
      possible to monomorphize ("template instantiate") functions so that each
      argument of !torch.nn.Module type corresponds to a unique instance.
      In practice, this limitation is either 1) (fundamental) due to truly
      dynamic use of modules, such as `m1 if cond() else m2` in Python code,
      or 2) (incidental) imprecision of the static analysis used in this pass
      which is used to calculate when a single intance is relevant. In general,
      this analysis is equivalent to the halting problem, but we can aim to
      improve this pass such that practical patterns are all handled.
      - Rationale: The fundamental limitation "1)" guarantees that the
        program can be lowered to a fixed set of globals without indirection
        across globals. In the absence of this property, most compiler
        analyses/transformations are significantly curtailed (or require very
        sophisticated implementations). For the moment, this restriction
        is deemed to be sufficiently reasonable to be a pragmatic choice to
        avoid front-loading the complexity of working with a representation that
        really does a good job of representing that kind of program.
        Additionally, it avoids front-loading the handling of programs which
        have !torch.nn.Module types at external calling convention boundaries.
    - All torch.nn_module's must be reachable by a unique path from the root
      - Rationale: Eliminates possibility of potentially exponential number of
        paths. Or worse, infinite number of paths when considering cyclic
        object graphs. Also as of Feb 2021, TorchScript won't import into
        this form (it has a bug related to the identity of submodules).
    - Two slots cannot have initial values that alias each other.
      - Rationale: This makes the representation of initial values simpler. Also
        as of Feb 2021, TorchScript won't import into this form except
        potentially for Tensors (it has a bug related to the identity of
        objects). And for tensors, the IValue importer only supports a
        very restricted form of aliasing anyway for other reasons. We are
        waiting for signals that more general handling of object aliasing is
        important to devote the effort to it.
  }];
}

def PrepareForGlobalizeObjectGraph
  : Pass<"torch-prepare-for-globalize-object-graph", "ModuleOp"> {
  let summary = "Lowering in preparation for globalizing";
  let constructor = "mlir::torch::Torch::createPrepareForGlobalizeObjectGraphPass()";
  let description = [{
    Establishes and the invariants needed by the
    torch-globalize-object-graph transformation. Fails if that cannot be
    accomplished.

    Currently, this just involves ensuring a small set of patterns have been
    applied.
  }];
}

def AdjustCallingConventions
  : Pass<"torch-adjust-calling-conventions", "ModuleOp"> {
  let summary = "Adjust the calling conventions of functions";
  let constructor = "mlir::torch::Torch::createAdjustCallingConventionsPass()";
  let description = [{
    Adjusts the calling conventions of functions in the module, with the aim of
    preparing them for backends and further lowering passes. As this changes
    the module calling convention, it should be considered a legalization
    step towards reaching IR that is suitable for an appropriate backend.
    All transformations are context-free and suitable for documenting
    at the user level if needed to clarify the eventual calling convention
    of compiled artifacts.
    This is not an optimization.

    The transformations performed are:
    - `torch.type_bound` annotations are incorporated into the type of the
      function arguments, which should be `!torch.tensor<...>`'s.
    - Python-isms are rewritten to MLIR-isms
      - NoneType return is rewritten to the absence of a return value.
      - Tuple return is rewritten to multiple return values.

  }];
}

def InlineGlobalSlots : Pass<"torch-inline-global-slots", "ModuleOp"> {
  let summary = "Inlines torch.global_slot ops.";
  let constructor = "mlir::torch::Torch::createInlineGlobalSlotsPass()";
  let description = [{
    Inlines torch.global_slot ops when it is safe to do so.

    Note: This pass inlines everything that is safe to inline. That is, it
    doesn't have a cost model. This is likely to pessimize programs with
    significant amounts of computation inside torch.initialize.global_slotsr
    regions (but this currently doesn't happen due to how TorchScript modules
    are imported -- the contents are just constants).
  }];
}

def ReduceOpVariants : Pass<"torch-reduce-op-variants", "func::FuncOp"> {
  let summary = "Reduces variants of ops to a smaller set of ops.";
  let constructor = [{
    mlir::torch::Torch::createReduceOpVariantsPass(/*extraLibrary=*/"")
  }];
  let options = [
    Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
           "MLIR module for verifying custom op value semantics">,
  ];
  let description = [{
    Replaces ops with other ops to reduce the number of variants that
    need to be handled elsewhere in the code.

    Examples of the transformations done in this pass are:
    - Convert operations with value semantics to operate on immutable tensors
    - Convert operations with in-place semantics (e.g. `add_`) or inherently
      mutable semantics (e.g. `add.out`) to their value-semantic equivalent.
    - Convert operations that involve a scalar promotion to the tensor
      variant plus a scalar promotion op.
  }];
}

def MaximizeValueSemantics : Pass<"torch-maximize-value-semantics", "func::FuncOp"> {
  let summary = "Use value-semantic tensors where possible.";
  let description = [{
    Use value-semantic tensors where possible to make the program more
    analyzable by later passes (also, backends prefer value semantics as well).

    This pass is analogous to an SSA-formation pass in a
    traditional compiler, with the added complication that arrays can alias
    each other in interesting ways.

    The current code doesn't implement any fancy algorithm, and is intended
    to be just sufficient for a first e2e spike. An algorithm inspired by the
    SSA formation literature will need to be implemented.

    Also, this pass doesn't currently handle interprocedural rewriting
    (of private functions), which is even more complex.
  }];
  let constructor = "mlir::torch::Torch::createMaximizeValueSemanticsPass()";
}


def RefinePublicReturn : Pass<"torch-refine-public-return", "ModuleOp"> {
  let summary = "Refine public return";
  let constructor = "mlir::torch::Torch::createRefinePublicReturnPass()";
  let description = [{
    Refines types of values returned from public functions based on
    intraprocedural information.

    This pass effectively encodes an assumption by the pass pipeline author that
    the public calling convention of the module can have its types refined,
    without causing ABI mismatches. This is frequently true -- for example, in
    many systems, `!torch.vtensor<[?,?],f32>`, `!torch.vtensor<[3,3],f32>` and
    `!torch.vtensor` are all the same data structure on calling
    convention boundaries.

    This pass is expected to run after shape refinement has occurred to
    otherwise resolve shapes, and is currently mainly useful to convert
    rank/dtype-erased function boundaries to ranked, dtyped code for
    compiler backends.

    This pass also changes the return to be a value tensor. This is incorrect
    in general because users may rely on the aliasing properties of non-value
    tensors, but for now it is deemed expedient to include this in this pass.
    TODO: Avoid hardcoding the value tensor assumption. In general, much
    as the type bound of an argument can be marked as having value semantics
    at the frontend level based on user concerns, so too should the returns
    from the function be annotated as having value semantics.
  }];
}

def DecomposeComplexOps : Pass<"torch-decompose-complex-ops", "func::FuncOp"> {
  let summary = "Decompose complicated torch operations";
  let constructor = [{
    mlir::torch::Torch::createDecomposeComplexOpsPass(/*legalOps=*/{})
  }];
  let options = [
    ListOption<"legalOps", "legal-ops", "std::string",
               "List of operation names that should be considered legal",
               "llvm::cl::ZeroOrMore">
  ];
  let description = [{
    Decompose torch operation that are losslessly represented as combinations of
    other operations, modulo appropropriate compiler fusion. Note that this pass
    is similar in spirit to ReduceOpVariants, but ReduceOpVariants is about
    systematic reductions of a large number of ops at once, guided mostly by
    traits.

    An example of the transformations done in this pass is:
    - convert aten.softmax to softmax(x, dim)
            => tmp=exp(x); tmp / sum(tmp, dim, keepdim=True)
  }];
}

def RecomposeComplexOps : Pass<"torch-recompose-complex-ops", "func::FuncOp"> {
  let summary = "Recompose torch operations that have been decomposed by TorchScript";
  let constructor = "mlir::torch::Torch::createRecomposeComplexOpsPass()";
  let description = [{
    There are certain ops that TorchScript will split into multiple ops that
    prevent optimizations in Torch-MLIR from taking place. In this pass such
    sequences of ops are identified and combined into a higher level op,
    preserving the original behavior, while allowing new optimizations to happen.

    An example is the handling of the indexing operation in PyTorch. The following

    ```
    input_tensor[1:2, :] = 7
    ```

    will get split into a series of `slice` ops to get the sub-tensor, then an
    in-place copy to overwrite the sub-tensor with the value 7. This type of
    pattern prevents the `MaximizeValueSemantics` pass from succeeding. So,
    using `RecomposeComplexOps`, the series of slices + copy is identified
    and turned into a single `index_put` operation.
  }];
}

def FuseQuantizedOps : Pass<"torch-fuse-quantized-ops", "func::FuncOp"> {
  let summary = "QDQ: Fuse recognized QDQ op sequences.";
  let constructor = "mlir::torch::Torch::createFuseQuantizedOpsPass()";
  let description = [{
    Torch models often represents quantized operations as the sequence:
      Dequantize
      DenseOp
      Quantize
    This allows the existing dense operations to be used without specifically
    representing quantized types. It is more computationally efficient to
    perform the dense operation in the quantized domain, so we fuse the
    quantization / dequantization behavior together and represent as purely
    quantized operations.
  }];
}

def MatchQuantizedCustomOps : Pass<"torch-match-quantized-custom-ops", "func::FuncOp"> {
  let summary = "Match quantized operations that occur in different namespace.";
  let constructor = "mlir::torch::Torch::createMatchQuantizedCustomOpsPass()";
  let description = [{
    Torch quantization utilities generated custom op versions of known aten
    quantziation operations. We can match these specially named operations and
    rewrite to the corresponding aten quantized operations.

    We handle this post import to maintain a simplified import process.
  }];
}

def ReifyShapeCalculations : Pass<"torch-reify-shape-calculations", "ModuleOp"> {
  let summary = "Reify shape calculations.";
  let constructor = [{
    mlir::torch::Torch::createReifyShapeCalculationsPass(/*extraLibrary=*/"")
  }];
  let options = [
    Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
           "MLIR module for splicing into the shape library">,
  ];
  let description = [{
  }];
}

def SimplifyShapeCalculations : Pass<"torch-simplify-shape-calculations", "func::FuncOp"> {
  let summary = "Simplify reified shape calculations.";
  let constructor = "mlir::torch::Torch::createSimplifyShapeCalculationsPass()";
  let description = [{
    Mainly for getting rank specialization.
  }];
}

def ReifyDtypeCalculations : Pass<"torch-reify-dtype-calculations", "ModuleOp"> {
  let summary = "Reify dtype calculations.";
  let constructor = [{
    mlir::torch::Torch::createReifyDtypeCalculationsPass(/*extraLibrary=*/"")
  }];
  let options = [
    Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
           "MLIR module for splicing into the dtype library">,
  ];
  let description = [{
  }];
}

def SimplifyDtypeCalculations : Pass<"torch-simplify-dtype-calculations", "func::FuncOp"> {
  let summary = "Simplify reified dtype calculations.";
  let constructor = "mlir::torch::Torch::createSimplifyDtypeCalculationsPass()";
  let description = [{
  }];
}

def DropAbstractInterpCalculations : Pass<"torch-drop-abstract-interp-calculations", "func::FuncOp"> {
  let summary = "Drop reified abstract interpretation calculations.";
  let constructor = "mlir::torch::Torch::createDropAbstractInterpCalculationsPass()";
  let description = [{
  }];
}

def EraseModuleInitializer
    : Pass<"torch-erase-module-initializer", "ModuleOp"> {
  let summary = "Erase the `torch.global_slot.module_initializer` op.";
  let constructor =
    "mlir::torch::Torch::createEraseModuleInitializerPass()";
  let description = [{
    Backends cannot currently handle module initializers, so we omit them from
    our backend contract. This pass removes the
    `torch.global_slot.module_initializer` op from the module if legal.
  }];
}

def LowerToBackendContract
    : Pass<"torch-lower-to-backend-contract", "ModuleOp"> {
  let summary = "Perform simplifications until the backend contract is satisfied.";
  let constructor = [{
    mlir::torch::Torch::createLowerToBackendContractPass(
      /*maxIterations=*/10, /*decompose=*/true, /*backendLegalOps=*/{}, /*extraLibrary=*/"")
  }];
  let description = [{
    This pass performs the bulk of the lowering of the program's computations
    to the backend contract. This pass does not do any global program
    restructuring -- it works entirely within a single semantic model
    of a `builtin.module` with `torch.global_slot` ops and `func.func` ops.
    
    This pass runs a set of simplifications within that semantic model until
    the backend contract is satisfied, and fails if it cannot be satisfied.
    In particular, the backend contract consists of:
    - Tensors
      - Have been converted to value semantics.
      - Have at least a known rank, though ideally a maximally inferred shape.
      - Have a known dtype.
    - `torch.global_slot`'s have been eliminated from the program.
    - Ops have been decomposed.

    This particular choice of backend contract was born out of a common set of
    requirements from backends, along with aligning with long-term PyTorch
    direction of being more tracing-based. The set of simplifications performed
    here can be thought of as simulating the kinds of simplifications that
    happen naturally as part of tracing, but in a way that is applicable
    to our TorchScript frontend. For the LazyTensorCore frontend, the backend
    contract trivially holds (except for certain decompositions).

    Generally it is not desirable to have a compiler where successful
    compilation depends on "optimizing hard enough", but in this case, there
    seems to be enough alignment and recognition in the industry that the
    Python-based programming model in the source program is too dynamic
    to feasibly handle in totality without a tracing approach that has access
    to the source program to re-trace in the face of dynamism (e.g. the ability
    to do what TorchDynamo calls "graph break"). We are attempting to maintain
    a practical compiler that works well given the current set of constraints
    of the TorchScript frontend that PyTorch provides us, and are working to
    co-design PyTorch's direction so that we land in a place where most of this
    "optimizing hard enough" is not necessary.
  }];
  let options = [
    Option<"maxIterations", "max-iterations", "int", /*default=*/"10",
           "Maximum number of invocations of the simplification pipeline.">,
    Option<"decompose", "decompose", "bool", /*default=*/"true",
           "Decompose ops.">,
    ListOption<"backendLegalOps", "backend-legal-ops", "std::string",
               "List of ops to be considered legal for the backend, such as 'aten.foo'.">,
    Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
           "MLIR module for splicing into the abstract interpretation library">,

  ];
  // TODO: Debug why this is needed, even though the input program has func.func
  // ops in it.
  let dependentDialects = ["func::FuncDialect"];
}

def VerifyBackendContractNoDecompositions
    : Pass<"torch-verify-backend-contract-no-decompositions", "ModuleOp"> {
  let summary = "Check that program satisfies backend contract.";
  let constructor = [{
    mlir::torch::Torch::createVerifyBackendContractNoDecompositionsPass()
  }];
  let description = [{
    This pass performs a set of inspections to check that program satisfies backend
    contract assuming that no decompositions were applied. In case of check failure
    it prints out the error message and returns `signalPassFailure()` status.
  }];
}

#endif // TORCHMLIR_TORCH_PASSES
