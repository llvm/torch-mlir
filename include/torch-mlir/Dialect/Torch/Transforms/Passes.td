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
      In pratice, this limitation is either 1) (fundamental) due to truly
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
      function arguments, which should be `!numpy.ndarray<...>`'s.
    - Python-isms are rewritten to MLIR-isms
      - NoneType return is rewritten to the absence of a return value.
      - Tuple return is rewritten to multiple return values.

  }];
}

def RefineTypes : Pass<"torch-refine-types", "func::FuncOp"> {
  let summary = "Refine types";
  let constructor = "mlir::torch::Torch::createRefineTypesPass()";
  let description = [{
    Refines types of the program. Currently, this means shapes and dtypes of
    tensors/arrays.
  }];
}

def InlineGlobalSlots : Pass<"torch-inline-global-slots", "ModuleOp"> {
  let summary = "Inlines torch.global_slot ops.";
  let constructor = "mlir::torch::Torch::createInlineGlobalSlotsPass()";
  let description = [{
    Inlines torch.global_slot ops when it is safe to do so.

    Note: This pass inlines everything that is safe to inline. That is, it
    doesn't have a cost model. This is likely to pessimize programs with
    significant amounts of computation inside torch.global_slot initializer
    regions (but this currently doesn't happen due to how TorchScript modules
    are imported -- the contents are just constants).
  }];
}

def ReduceOpVariants : Pass<"torch-reduce-op-variants", "func::FuncOp"> {
  let summary = "Reduces variants of ops to a smaller set of ops.";
  let constructor = "mlir::torch::Torch::createReduceOpVariantsPass()";
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
  let constructor = "mlir::torch::Torch::createDecomposeComplexOpsPass()";
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

def DecomposeComplexOpsEarly : Pass<"torch-decompose-complex-ops-early", "func::FuncOp"> {
  let summary = "Decompose complicated torch operations before shape inference";
  let constructor = "mlir::torch::Torch::createDecomposeComplexOpsEarlyPass()";
  let description = [{
    Decompose torch operation that are losslessly represented as combinations of
    other operations, modulo appropropriate compiler fusion before shape inference.

    An example of the transformations done in this pass is:
    - convert aten.softmax to softmax(x, dim)
            => tmp=exp(x); tmp / sum(tmp, dim, keepdim=True)
  }];
}

def ReifyShapeCalculations : Pass<"torch-reify-shape-calculations", "ModuleOp"> {
  let summary = "Decompose complicated torch operations";
  let constructor = "mlir::torch::Torch::createReifyShapeCalculationsPass()";
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

def DropShapeCalculations : Pass<"torch-drop-shape-calculations", "func::FuncOp"> {
  let summary = "Drop reified shape calculations.";
  let constructor = "mlir::torch::Torch::createDropShapeCalculationsPass()";
  let description = [{
  }];
}

#endif // TORCHMLIR_TORCH_PASSES
