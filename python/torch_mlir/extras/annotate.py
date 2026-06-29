# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Helpers for annotating a `torch.export.ExportedProgram` with MLIR attributes.

The fx_importer reads two reserved `torch.fx.Node.meta` keys on the
post-decomposition graph:

  * `node.meta["mlir.arg_attrs"]` on a placeholder - MLIR func arg attrs
  * `node.meta["mlir.attrs"]` on any node          - MLIR op attrs

These helpers are thin conveniences that resolve FX nodes by semantic identity
(user-input name, submodule fully-qualified name) and write into those dicts.
They carry no dialect-specific vocabulary: the user chooses the attribute names.

Typical usage, as the `annotate=` callback of `torch_mlir.fx.export_and_import`:

    def annotate(prog):
        annotate_arg(prog, "x", {"my.attr": True})
        annotate_module(prog, "act1", {"my.attr_lo": -20.0, "my.attr_hi": 20.0})

    mlir = torch_mlir.fx.export_and_import(model, sample, annotate=annotate, ...)

Callback-style invocation ensures the writes land *after* `run_decompositions`,
which otherwise strips custom `node.meta` entries. Users who run decomposition
themselves can call these directly on the decomposed program and pass
`decomposition_table={}`.
"""

from typing import Any, Dict, Union

import torch.fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from .fx_importer import MLIR_ARG_ATTRS_META_KEY, MLIR_OP_ATTRS_META_KEY


__all__ = ["annotate_arg", "annotate_module"]


def annotate_arg(
    prog: ExportedProgram,
    name_or_index: Union[str, int],
    attrs: Dict[str, Any],
) -> None:
    """Attach MLIR arg attrs to a user input of `prog`.

    `name_or_index` selects the user input by `forward`-parameter name (str)
    or by positional index among user inputs (int). `attrs` are Python values
    coerced to MLIR `Attribute`s by the importer; pass a pre-built
    `mlir.ir.Attribute` for custom encodings.
    """
    user_names = [
        s.arg.name for s in prog.graph_signature.input_specs
        if s.kind == InputKind.USER_INPUT
    ]
    if isinstance(name_or_index, bool) or not isinstance(name_or_index, (str, int)):
        raise TypeError(
            f"name_or_index must be str or int, got {type(name_or_index).__name__}"
        )
    if isinstance(name_or_index, int):
        if not 0 <= name_or_index < len(user_names):
            raise IndexError(
                f"user input index {name_or_index} out of range 0..{len(user_names)}"
            )
        target = user_names[name_or_index]
    else:
        if name_or_index not in user_names:
            raise KeyError(
                f"{name_or_index!r} is not a user input of this program; "
                f"user inputs are {user_names}"
            )
        target = name_or_index
    for n in prog.graph.nodes:
        if n.op == "placeholder" and n.name == target:
            n.meta.setdefault(MLIR_ARG_ATTRS_META_KEY, {}).update(attrs)
            return
    raise KeyError(f"placeholder '{target}' not found in FX graph")


def annotate_module(
    prog: ExportedProgram,
    fqn: str,
    attrs: Dict[str, Any],
    *,
    mode: str = "last",
) -> int:
    """Attach MLIR op attrs to FX nodes belonging to `nn.Module` at `fqn`.

    Matches on `node.meta["nn_module_stack"]`, which `torch.export` populates
    with the submodule path that produced each node. For a module like
    `model.act1`, pass `fqn="act1"` (suffix match) or `fqn="model.act1"`
    (exact match).

    `mode`:
      * `"last"` (default): attach to the last matching node only. Right for
        activation-style submodules where a single representative op carries
        the annotation downstream.
      * `"all"`: attach to every matching node.

    Returns the number of nodes annotated.
    """
    if mode not in ("last", "all"):
        raise ValueError(f"mode must be 'last' or 'all', got {mode!r}")
    matches = []
    for n in prog.graph.nodes:
        stack = n.meta.get("nn_module_stack") or {}
        for entry_fqn, _ in stack.values():
            if entry_fqn == fqn or entry_fqn.endswith("." + fqn):
                matches.append(n)
                break
    if not matches:
        return 0
    targets = [matches[-1]] if mode == "last" else matches
    for n in targets:
        n.meta.setdefault(MLIR_OP_ATTRS_META_KEY, {}).update(attrs)
    return len(targets)
