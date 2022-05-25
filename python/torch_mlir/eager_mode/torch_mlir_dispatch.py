# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
from __future__ import annotations

from typing import Any, Callable, Tuple
from typing import Dict

import torch
from torch.fx import immutable_collections
from torch.fx.operator_schemas import (
    _torchscript_schema_to_signature,
    _args_kwargs_to_normalized_args_kwargs,
)
from torch_mlir._mlir_libs._jit_ir_importer import get_registered_ops

from torch_mlir.dialects import torch as torch_dialect

OP_REGISTRY = {op["name"]: op for op in get_registered_ops()}
SUPPORTED_OPS = frozenset(
    [
        member.OPERATION_NAME
        for member in vars(torch_dialect).values()
        if hasattr(member, "OPERATION_NAME")
    ]
)


class UnsupportedByTorchMlirEagerMode(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


def normalize_args_kwargs(target: Callable, args: Tuple[Any], kwargs: Dict[str, Any]):
    """Fill in default values for optional args, which are dependent on the schema."""
    sig = _torchscript_schema_to_signature(target._schema)
    _, new_kwargs = _args_kwargs_to_normalized_args_kwargs(
        sig, args, kwargs, normalize_to_only_use_kwargs=True
    )
    if "self" in new_kwargs:
        new_kwargs["input"] = new_kwargs.pop("self")

    # Flatten lists of args for ops that takes lists, such as torch.cat.
    to_remove = set()
    to_add = {}
    for k, v in new_kwargs.items():
        if isinstance(v, (tuple, list)) and len(v) and isinstance(v[0], torch.Tensor):
            to_remove.add(k)
            for i, vv in enumerate(v):
                to_add[f"{k}_flattened_{i}"] = vv

    for rem in to_remove:
        del new_kwargs[rem]
    new_kwargs.update(**to_add)

    # Sort here in order to have consistency across TS graph and
    # MLIR module.
    sorted_kwargs = dict(sorted(new_kwargs.items()))
    return immutable_collections.immutable_dict(sorted_kwargs)


def get_registered_op(op):
    registered_op = OP_REGISTRY[(op._schema.name, op._schema.overload_name)]
    return registered_op


def check_get_aliased_arg(func: Callable,):
    """Write back to mutable args that aren't properly handled otherwise.

    Because of how we pass values to the backend we don't currently support ops that mutate operands.
    That includes both inplace variants and outplace variants. Additionally, Torch-MLIR,
    as of right now, only handles arguments with value semantics, so we need to manually fake those semantics, which
    we can for these special cases. Hence, the solution is to manually write back to the same operand that the
    conventional pytorch op variant would write to.

    Note that there are ops where multiple operands are mutable (such as batchnorm outplace variants that
    mutate running_mean and running_var). We don't currently handle those.
    """

    registered_op = get_registered_op(func)
    if not registered_op["is_mutable"]:
        return None

    if len(registered_op["returns"]) > 1:
        raise UnsupportedByTorchMlirEagerMode(
            "TorchMLIR doesn't handle multiple aliased returns yet."
        )

    aliased_arg = next(
        arg
        for arg in registered_op["arguments"]
        if "alias_info" in arg and arg["alias_info"]["is_write"]
    )
    assert (
        "alias_info" in registered_op["returns"][0]
        and registered_op["returns"][0]["alias_info"]["is_write"]
        and len(registered_op["returns"][0]["alias_info"]["after"]) == 1
        and registered_op["returns"][0]["alias_info"]["after"][0]
    )
    assert (
        len(aliased_arg["alias_info"]["after"]) == 1
        and aliased_arg["alias_info"]["after"][0]
        == registered_op["returns"][0]["alias_info"]["after"][0]
    )

    return aliased_arg["name"] if aliased_arg["name"] != "self" else "input"
