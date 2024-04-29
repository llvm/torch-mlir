# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import pdb

# This file implements a pure-Python importer from a restricted subset of
# FX IR into MLIR.
#
# As described in the
# [long-term roadmap](https://github.com/llvm/torch-mlir/blob/main/docs/long_term_roadmap.md#refactoring-the-frontend),
# the goal is to import directly in the Torch-MLIR backend contract by
# using the available PyTorch infra for doing functionalization,
# shape inference, etc. Thus, this importer imports a very specific subset
# of the possible FX IR that is co-designed with the PyTorch infra that produces
# the FX graph -- see the `torch_mlir.dynamo` module for that, and see the
# `_verify_fx_graph_conforms_to_subset` function for the operational definition.
#
# In fact, because of the generality of FX IR (e.g. the use of raw Python
# callables as node.target), there is really no well-defined way to implement a
# general FX -> MLIR importer. Reuse or extension of this code for other
# FX -> MLIR use cases should be done carefully, and likely will involve
# introducing new concepts or abstractions into the import process.

from typing import Dict, Tuple
from typing_extensions import deprecated

import operator
import re

import torch

import torch_mlir.ir as ir
import torch_mlir.dialects.func as func_dialect
import torch_mlir.dialects.torch as torch_dialect


def _is_valid_meta_val(val):
    # We currently allow only FakeTensor's or lists of FakeTensor's
    # as meta['val']. However, this can potentially change also hold a SymInt
    # in the future. See:
    # https://github.com/pytorch/pytorch/issues/90839#issuecomment-1352856661
    if isinstance(val, torch._subclasses.FakeTensor):
        return True
    if isinstance(val, (tuple, list)):
        return all(isinstance(x, torch._subclasses.FakeTensor) for x in val)
    return False


def _verify_fx_graph_conforms_to_subset(g: torch.fx.Graph):
    # TODO: Report errors with source locations if possible.
    def _check_meta_val(node):
        if "val" not in node.meta:
            raise Exception(f"Unsupported: missing node.meta['val']: {node}")
        if not _is_valid_meta_val(node.meta["val"]):
            raise Exception(
                f"Unsupported: node.meta['val'] is not a FakeTensor or list of FakeTensor's: {node}; {node.meta['val']}"
            )

    for node in g.nodes:
        if node.op not in ("placeholder", "call_function", "output"):
            raise Exception(f"Unsupported op: {node.op}")
        if node.op == "placeholder":
            _check_meta_val(node)
        if node.op == "call_function":
            _check_meta_val(node)
            # We only support OpOverload for computations because the `torch`
            # dialect ops model the full qualified op name, including overload.
            # We also support operator.getitem because that is how multiple
            # results are modeled.
            if isinstance(node.target, torch._ops.OpOverload):
                for type_ in (r.type for r in node.target._schema.returns):
                    if isinstance(type_, torch.TensorType):
                        continue
                    raise Exception(
                        f"Unsupported: return type {type_} in schema for {node.target}"
                    )
                if len(node.args) != len(node.target._schema.arguments):
                    assert len(node.args) < len(node.target._schema.arguments)
                    for i, argument in enumerate(
                        node.target._schema.arguments[len(node.args) :]
                    ):
                        if (
                            not argument.has_default_value()
                            and argument.name not in node.kwargs
                        ):
                            raise Exception(
                                f"Unsupported: missing default value for argument {i} in schema for {node.target}"
                            )
                continue
            if node.target is operator.getitem:
                continue
            raise Exception(f"Unsupported call_function target: {node.target}")


# ==============================================================================
# Type import
# ==============================================================================


def _torch_type_to_mlir_type_string(t: torch.Type) -> str:
    # This is weird -- for Node's, since they are untyped, we use the
    # node.meta['val'] to get the type (which is a tensor type with sizes and
    # dtype).
    # But for things that are associated with a schema, we use the schema to get
    # the type. This creates problems for things like a list<tensor> because
    # then we don't have sizes or dtypes available.
    if isinstance(t, torch.ListType):
        return f"list<{_torch_type_to_mlir_type_string(t.getElementType())}>"
    if isinstance(t, torch.BoolType):
        return "bool"
    if isinstance(t, torch.IntType):
        return "int"
    if isinstance(t, torch.FloatType):
        return "float"
    if isinstance(t, torch.StringType):
        return "string"
    if isinstance(t, torch.TensorType):
        return "vtensor"
    if isinstance(t, torch.OptionalType):
        return f"optional<{_torch_type_to_mlir_type_string(t.getElementType())}>"
    raise Exception(f"Unsupported type: {t}")


def _torch_type_to_mlir_type(t: torch.Type):
    return ir.Type.parse(f"!torch.{_torch_type_to_mlir_type_string(t)}")


def _convert_dtype_to_mlir_type(dtype: torch.dtype) -> str:
    # See the table in TorchTypes.td:AnyTorchTensorType's documentation.
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float64:
        return "f64"
    if dtype == torch.uint8:
        return "ui8"
    if dtype == torch.int8:
        return "si8"
    if dtype == torch.int16:
        return "si16"
    if dtype == torch.int32:
        return "si32"
    if dtype == torch.int64:
        return "si64"
    if dtype == torch.bool:
        return "i1"
    if dtype == torch.qint8:
        return "!torch.qint8"
    if dtype == torch.quint8:
        return "!torch.quint8"
    if dtype == torch.complex64:
        return "complex<f32>"
    if dtype == torch.complex128:
        return "complex<f64>"

    raise Exception(f"Unsupported dtype: {dtype}")


def _import_fake_tensor_as_mlir_type(
    fake_tensor: torch._subclasses.FakeTensor,
) -> ir.Type:
    # TODO: Find story for how to get dynamically shaped tensors here.
    shape = ",".join(str(d) for d in fake_tensor.shape)
    dtype = _convert_dtype_to_mlir_type(fake_tensor.dtype)
    return ir.Type.parse(f"!torch.vtensor<[{shape}],{dtype}>")


def _mlir_types_for_node(node: torch.fx.Node) -> ir.Type:
    if isinstance(node.meta["val"], (tuple, list)):
        return [_import_fake_tensor_as_mlir_type(v) for v in node.meta["val"]]
    return [_import_fake_tensor_as_mlir_type(node.meta["val"])]


def _extract_function_type_from_graph(g: torch.fx.Graph) -> ir.FunctionType:
    input_types = []
    for node in g.nodes:
        if node.op == "placeholder":
            input_types.append(_mlir_types_for_node(node)[0])
        if node.op == "output":
            # TODO(DNS): Test this or add verifier that it can't happen.
            result_types = torch.fx.map_arg(
                node.args[0], lambda n: _mlir_types_for_node(n)[0]
            )
    # Note: We import directly to the backend contract -- multiple results
    # are modeled with func.func native multiple results rather than as a
    # singleton value / tuple.
    return ir.FunctionType.get(input_types, result_types)


# ==============================================================================
# FX Graph import
# ==============================================================================

DTYPE_TO_INT = {
    # TODO(DNS): Fill in from AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    # torch.complex_half 8
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.qint8: 12,
    torch.quint8: 13,
    # torch.qint32 14
    torch.bfloat16: 15,
}

MEMORY_FORMAT_TO_INT = {
    # https://github.com/pytorch/pytorch/c10/core/MemoryFormat.h#L28
    torch.contiguous_format: 0,
    torch.preserve_format: 1,
    torch.channels_last: 2,
    torch.channels_last_3d: 3,
}

LAYOUT_TO_INT = {
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_layouts.cpp
    torch.strided: 0,
    torch.sparse_coo: 1,
    torch.sparse_csr: 2,
    torch.sparse_csc: 3,
    torch.sparse_bsr: 4,
    torch.sparse_bsc: 5,
}


def _mlir_location_for_node(node: torch.fx.Node) -> ir.Location:
    stack_trace = node.stack_trace
    if stack_trace is None:
        return ir.Location.unknown()
    # TODO: Avoid needing to regex match this.
    # https://github.com/pytorch/pytorch/issues/91000
    m = re.search(r"""File "([^"]+)", line ([0-9]+),""", node.stack_trace)
    filename, line = m.group(1), int(m.group(2))
    return ir.Location.file(filename, line, col=0)


class _FXGraphImporter:
    def __init__(self, g: torch.fx.Graph, func_name: str):
        self._g = g
        self._func_name = func_name
        # For each node, we track a mapping to MLIR Value's.
        # Technically all Node's have a single output (which can be a tuple of
        # values in case of multiple returns), but we treat them as having
        # multiple returns directly. This matches how the Node's
        # node.meta['val'] is set up, since it contains a list with multiple
        # FakeTensor's in case of a tuple return with multiple elements.
        self._env: Dict[Tuple[torch.fx.Node, int], ir.Value] = {}
        self._module = ir.Module.create(ir.Location.unknown())
        self._module.operation.attributes["torch.debug_module_name"] = (
            ir.StringAttr.get(func_name)
        )
        function_type = _extract_function_type_from_graph(g)
        func = func_dialect.FuncOp(
            func_name,
            function_type,
            loc=ir.Location.unknown(),  # TODO: Can we do better?
            ip=ir.InsertionPoint(self._module.body),
        )
        self._body_block = ir.Block.create_at_start(func.body, function_type.inputs)

    def import_graph(self) -> ir.Module:
        with ir.InsertionPoint(self._body_block):
            num_placeholders_seen = 0
            for node in self._g.nodes:
                with _mlir_location_for_node(node):
                    if node.op == "placeholder":
                        self._env[(node, 0)] = self._body_block.arguments[
                            num_placeholders_seen
                        ]
                        num_placeholders_seen += 1
                    if node.op == "call_function":
                        if node.target is operator.getitem:
                            self._env[(node, 0)] = self._env[
                                (node.args[0], node.args[1])
                            ]
                        else:
                            self._import_op_overload_call(node)
                    if node.op == "output":
                        # Note that the output node is a singleton tuple holding
                        # a tuple of return values (without the single-element special
                        # case)
                        # DNS: Test or verify no literals as results.
                        operands = [self._import_argument(arg) for arg in node.args[0]]
                        func_dialect.ReturnOp(operands)
        return self._module

    def _import_op_overload_call(self, node: torch.fx.Node):
        assert node.op == "call_function"
        assert isinstance(node.target, torch._ops.OpOverload)
        schema = node.target._schema

        # Extract the `torch` dialect op name.
        namespace, _, unqualified_name = schema.name.partition("::")
        mlir_op_name = f"torch.{namespace}.{unqualified_name}"
        if schema.overload_name != "":
            mlir_op_name += f".{schema.overload_name}"

        # DNS: Unregistered ops
        assert ir.Context.current.is_registered_operation(
            mlir_op_name
        ), f"Unregistered operation: {mlir_op_name}"

        # Construct the Operation.
        result_types = _mlir_types_for_node(node)
        operands = []
        # `schema.arguments` is a bit confusing in this context, since
        # `Argument` is the term that FX uses analogous to mlir "Value". It is
        # more precise to call them "formal parameters".
        for i, parameter in enumerate(node.target._schema.arguments):
            if parameter.kwarg_only and parameter.name in node.kwargs:
                arg = node.kwargs[parameter.name]
            elif i < len(node.args):
                arg = node.args[i]
            else:
                arg = parameter.default_value
            operands.append(self._import_argument(arg, parameter.type))
        operation = ir.Operation.create(
            mlir_op_name,
            results=result_types,
            operands=operands,
        )
        for i, value in enumerate(operation.results):
            self._env[(node, i)] = value

    def _import_argument(
        self, arg: torch.fx.node.Argument, expected_type_for_literal=None
    ) -> ir.Value:
        """Import an FX `Argument`, which is analogous to an MLIR `Value`.

        Args:
            arg: The FX `Argument` to import.
            expected_type_for_literal: If `arg` is a literal (such as a Python
              `int` or `float` object), this is the expected JIT IR type. This
              allows disambiguating certain cases, such as importing an optional
              type.
        Returns:
            The imported MLIR `Value`.
        """
        if isinstance(arg, torch.fx.Node):
            return self._env[(arg, 0)]
        assert expected_type_for_literal is not None
        return self._import_literal(arg, expected_type_for_literal)

    def _import_literal(self, arg: torch.fx.node.Argument, expected_type) -> ir.Value:
        if arg is None:
            return torch_dialect.ConstantNoneOp().result
        if isinstance(expected_type, torch.OptionalType):
            return self._import_argument(arg, expected_type.getElementType())
        if isinstance(arg, bool):
            return torch_dialect.ConstantBoolOp(
                ir.IntegerAttr.get(ir.IntegerType.get_signless(1), arg)
            ).result
        if isinstance(arg, int):
            return torch_dialect.ConstantIntOp(
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), arg)
            ).result
        if isinstance(arg, float):
            return torch_dialect.ConstantFloatOp(ir.FloatAttr.get_f64(arg)).result
        if isinstance(arg, str):
            return torch_dialect.ConstantStrOp(ir.StringAttr.get(arg)).result
        if isinstance(arg, torch.dtype):
            assert isinstance(expected_type, torch.IntType)
            return self._import_argument(DTYPE_TO_INT[arg], expected_type)
        if isinstance(arg, torch.device):
            # TODO(DNS): Device index? arg.index
            return torch_dialect.ConstantDeviceOp(ir.StringAttr.get(arg.type)).result
        if isinstance(arg, torch.memory_format):
            assert isinstance(expected_type, torch.IntType)
            return self._import_argument(MEMORY_FORMAT_TO_INT[arg], expected_type)
        if isinstance(arg, torch.layout):
            assert isinstance(expected_type, torch.IntType)
            return self._import_argument(LAYOUT_TO_INT[arg], expected_type)
        if isinstance(arg, list):
            assert isinstance(expected_type, torch.ListType)
            element_type = expected_type.getElementType()
            if isinstance(element_type, torch.TensorType):
                assert all(
                    torch.fx.node.map_aggregate(
                        arg, lambda a: _is_valid_meta_val(a.meta.get("val"))
                    )
                )
                els = [self._env[e, 0] for e in arg]

            else:
                element_type = _torch_type_to_mlir_type(element_type)
                els = [self._import_argument(e, element_type) for e in arg]

            # import pydevd_pycharm
            # pydevd_pycharm.settrace('localhost', port=8888, stdoutToServer=True, stderrToServer=True)
            return torch_dialect.PrimListConstructOp(
                _torch_type_to_mlir_type(expected_type),
                els,
            ).result
        raise Exception(f"Unsupported literal: {arg}")


@deprecated("Please use fx importer as a replacement to support torchdynamo")
def import_fx_graph_as_func(g: torch.fx.Graph, func_name: str) -> ir.Module:
    """Imports the given FX graph as a function in a new MLIR module.

    Args:
        g: The FX graph to import.
        func_name: The sym_name of the `func.func` to import the graph into.
    Returns:
        A new MLIR module containing the imported function.
    """
    # Note that this function imports a fx.Graph instead of an fx.GraphModule.
    # The reason is that the supported subset only involves stateless
    # fx.Graph's, so the state held on the fx.GraphModule is not necessary.
    _verify_fx_graph_conforms_to_subset(g)
    with ir.Context() as context, ir.Location.unknown(context=context):
        torch_dialect.register_dialect(context)
        return _FXGraphImporter(g, func_name).import_graph()
