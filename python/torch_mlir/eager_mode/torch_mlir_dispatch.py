# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
from __future__ import annotations

from typing import Any, Callable, Tuple
from typing import Dict

from torch.fx import immutable_collections
from torch.fx.node import map_aggregate
from torch.fx.operator_schemas import normalize_function, create_type_hint
from torch_mlir._mlir_libs._jit_ir_importer import (
    get_registered_ops,
)  # pytype: disable=import-error

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

    arg_types = map_aggregate(args, type)
    assert isinstance(arg_types, tuple)
    arg_types = map_aggregate(map_aggregate(args, type), create_type_hint)
    kwarg_types = {
        k: create_type_hint(map_aggregate(v, type)) for k, v in kwargs.items()
    }

    new_args_and_kwargs = normalize_function(
        target.op,
        args,
        kwargs,
        arg_types,
        kwarg_types,
        normalize_to_only_use_kwargs=True,
    )
    assert new_args_and_kwargs, "Couldn't normalize args and kwargs"
    _, new_kwargs = new_args_and_kwargs
    return immutable_collections.immutable_dict(new_kwargs)


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
    return aliased_arg["name"]
