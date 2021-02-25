//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TORCH_PASSES
#define NPCOMP_TORCH_PASSES

include "mlir/Pass/PassBase.td"

def GlobalizeObjectGraph : Pass<"torch-globalize-object-graph", "ModuleOp"> {
  let summary = "Converts TorchScript object graphs to a globalized form";
  let constructor = "mlir::NPCOMP::Torch::createGlobalizeObjectGraphPass()";
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
    fixed set of globals and functions.

    Of course, only a subset of programs can be transformed, and this pass fails
    with an error if the conditions are violated.

    Specifically, the restrictions are:
    - There must be a unique torch.nn_module that is not the value of a slot
      of any other torch.nn_module
      - Rationale: Allows us to have a notion of a unique "root" op, which is
        used to define linkage. This also matches how TorchScript imports in
        practice (`torch.jit.script` imports a single root object).
    - There must be exactly one instance of each torch.class_type. Equivalently,
      Every torch.nn_module must have a distinct type.
      - Rationale: This guarantee precludes things like selecting between
        multiple modules dynamically at runtime, which would require indirecting
        between the separate storage of each instance.
    - All torch.nn_module's must be reachable by a unique path from the root
      - Rationale: Eliminates possibility of potentially exponential number of
        paths. Or worse, infinite number of paths when considering cyclic
        object graphs. Also as of Feb 2021, TorchScript won't import into
        this form (it has a bug related to the identity of submodules).
    - Two slots cannot have initial values that alias each other.
      - Rationale: This makes the representation of initial values simpler. Also
        as of Feb 2021, TorchScript won't import into this form except
        potentially for Tensors (it has a bug related to the identity of
        objects). And for tensors, the npcomp IValue importer only supports a
        very restricted form of aliasing anyway for other reasons. We are
        waiting for signals that more general handling of object aliasing is
        important to devote the effort to it.
  }];
}

#endif // NPCOMP_TORCH_PASSES
