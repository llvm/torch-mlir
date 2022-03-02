# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any, Callable, Tuple, Union
from typing import List, Dict

import numpy as np
import torch
import torch._C
from torch.fx.node import map_aggregate
from torch.fx.operator_schemas import normalize_function, create_type_hint
from torch.utils._pytree import tree_map
from torch_mlir._mlir_libs._mlir.passmanager import PassManager

from torch_mlir.dialects import torch as torch_dialect
from torch_mlir._mlir_libs._jit_ir_importer import get_registered_ops # pytype: disable=import-error
from torch_mlir.eager_mode.ir_building import build_module, TorchTensorType

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


def check_supported_op(schema: torch._C.FunctionSchema) -> bool:
    return (
        "torch."
        + schema.name.replace("::", ".")
        + ("." + schema.overload_name if schema.overload_name else "")
    ) in SUPPORTED_OPS


def is_tensor_type(typ: torch._C.Type):
    return typ.isSubtypeOf(torch.TensorType.get()) or (
        isinstance(typ, torch.OptionalType)
        and typ.getElementType().isSubtypeOf(torch._C.TensorType.get())
    )


def normalize_args_kwargs(target: Callable, args: Tuple[Any], kwargs: Dict[str, Any]):
    """Fill in default values for optional args, which are dependent on the schema."""

    arg_types = map_aggregate(args, type)
    assert isinstance(arg_types, tuple)
    arg_types = tuple([create_type_hint(i) for i in arg_types])
    kwarg_types = {k: type(v) for k, v in kwargs.items()}

    new_args_and_kwargs = normalize_function(
        target, args, kwargs, arg_types, kwarg_types, normalize_to_only_use_kwargs=False
    )
    assert new_args_and_kwargs, "Couldn't normalize args and kwargs"
    new_args, new_kwargs = new_args_and_kwargs
    return new_args, new_kwargs


def build_script_function(
    schema: torch._C.FunctionSchema,
    args: List[torch._C.Argument],
    kwargs: Dict[str, Any],
) -> torch.jit.ScriptFunction:
    """Build a torch.jit.ScriptFunction that corresponds to the schema.

    Constants are inlined for the purposes of invalidating the compile cache when they change.
    """

    # Creates empty TS graph.
    graph = torch._C.Graph()
    # Creates and inserts node with identifier `schema.name`; NB node has no inputs or outputs at this point.
    node = graph.insertNode(graph.create(schema.name, len(schema.returns)))
    # Associate graph inputs/outputs with node inputs/outputs.
    for i, arg in enumerate(schema.arguments):
        # Find value corresponding to schema arg, either in positional or kw args.
        kwarg = False
        if arg.name in kwargs:
            val = kwargs[arg.name]
            kwarg = True
        else:
            val = args[i]

        # If arg is a tensor, then add input to the graph corresponding to arg.
        if is_tensor_type(arg.type) and val is not None:
            inp = graph.addInput()
            if isinstance(arg.type, torch.OptionalType):
                inp.setType(arg.type.getElementType())
            else:
                inp.setType(arg.type)

            if kwarg:
                # Rename for debugging aid.
                inp.setDebugName(arg.name)
        # If arg is a constant, inline (at the top of the graph).
        else:
            inp = graph.insertConstant(val)
            inp.node().moveBefore(node)

        node.addInput(inp)

    if node.hasMultipleOutputs():
        for outp in node.outputs():
            graph.registerOutput(outp)
    else:
        graph.registerOutput(node.output())

    fn_name = str(node).strip()
    fn = torch._C._create_function_from_graph(fn_name, graph)
    return fn


def annotate_args_kwargs(
    script_fun: torch._C.ScriptFunction,
    normalized_args: List[Any],
    normalized_kwargs: Dict[str, Any],
):
    unwrapped_normalized_args = tree_map(
        lambda x: x.detach().contiguous().numpy() if isinstance(x, torch.Tensor) else x,
        normalized_args,
    )
    unwrapped_normalized_kwargs = tree_map(
        lambda x: x.detach().contiguous().numpy() if isinstance(x, torch.Tensor) else x,
        normalized_kwargs,
    )

    annotations = []
    tensor_args = []
    for i, arg in enumerate(unwrapped_normalized_args):
        if isinstance(arg, np.ndarray):
            # TODO: Remove once size zero dimensions are handled by torch-mlir.
            shape = tuple(map(lambda x: x or -1, arg.shape))
            annotations.append(
                TorchTensorType(shape=shape, dtype=normalized_args[i].dtype)
            )
            tensor_args.append(arg)

    # Pull out tensor kwargs and put them in positional order.
    tensor_kwargs_flat = []
    if unwrapped_normalized_kwargs:
        tensor_kwargs = {}
        arg_idxs = {
            arg_name: i
            for i, arg_name in enumerate(
                [arg.name for arg in script_fun.schema.arguments]
            )
        }
        for i, (kw, arg) in enumerate(unwrapped_normalized_kwargs.items()):
            if isinstance(arg, np.ndarray):
                tensor_kwargs[arg_idxs[kw]] = (arg, normalized_kwargs[kw].dtype)

        for i in range(len(tensor_kwargs)):
            arg, arg_dtype = tensor_kwargs[i]
            annotations.append(TorchTensorType(shape=tuple(arg.shape), dtype=arg_dtype))
            tensor_kwargs_flat.append(arg)

    return annotations, tensor_args, tensor_kwargs_flat


def write_back_to_mutable(
    registered_op: Dict,
    out: Union[np.ndarray, List[np.ndarray]],
    all_tensor_args: List[np.ndarray],
):
    """Write back to mutable args that aren't properly handled otherwise.

    Because of how we pass values to the backend (by copying the tensor to a numpy array) we don't currently support
    ops that mutate operands. That includes both inplace variants and outplace variants. Additionally, Torch-MLIR,
    as of right now, only handles arguments with value semantics, so we need to manually fake those semantics, which
    we can for these special cases. Hence, the solution is to manually write back to the same operand that the
    conventional pytorch op variant would write to.

    Note that there are ops where multiple operands are mutable (such as batchnorm outplace variants that
    mutate running_mean and running_var). We don't currently handle those.
    """
    if len(registered_op["returns"]) > 1:
        raise UnsupportedByTorchMlirEagerMode(
            "TorchMLIR doesn't handle multiple aliased returns yet."
        )
    else:
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
        np.copyto(all_tensor_args[0], out)

    return out


def try_torch_mlir_eager(op, args, kwargs, backend):
    if hasattr(op, "op_name"):
        op_name = op.op_name
    elif hasattr(op, "__name__"):
        # Handle builtin_function_or_method.
        op_name = op.__name__
    else:
        raise RuntimeError(f"op {op} has no name")

    if op_name == "detach":
        # We don't handle detach as it only pertains to autograd graph construction, which is handled by pytorch.
        raise UnsupportedByTorchMlirEagerMode("detaching")

    if not hasattr(op, "_schema"):
        raise RuntimeError(f"op {op} has no schema.")

    new_args, new_kwargs = normalize_args_kwargs(op.overloadpacket, args, kwargs)

    if "layout" in new_kwargs and new_kwargs["layout"] not in {0, None}:
        raise UnsupportedByTorchMlirEagerMode(
            f"{new_kwargs['layout']} layout not supported."
        )
    if "memory_format" in new_kwargs and new_kwargs["memory_format"] not in {0, None}:
        raise UnsupportedByTorchMlirEagerMode(
            f"{new_kwargs['memory_format']} memory format not supported."
        )

    script_fun = build_script_function(op._schema, new_args, new_kwargs)
    annotations, np_tensor_args, np_tensor_kwargs_flat = annotate_args_kwargs(
        script_fun, new_args, new_kwargs
    )

    eager_module = build_module(script_fun, annotations)
    with eager_module.context:
        pm = PassManager.parse(
            "torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline"
        )
        pm.run(eager_module)
    compiled_module = backend.compile(eager_module)
    loaded_module = backend.load(compiled_module)
    op_mlir_backend_callable = getattr(loaded_module, script_fun.name)
    assert (
        op_mlir_backend_callable is not None
    ), f"Couldn't find function {script_fun.name} in module."

    all_tensor_args = np_tensor_args + np_tensor_kwargs_flat
    out = op_mlir_backend_callable(*all_tensor_args)

    registered_op = OP_REGISTRY[(op._schema.name, op._schema.overload_name)]
    if registered_op["is_mutable"]:
        out = write_back_to_mutable(registered_op, out, all_tensor_args)

    return out
