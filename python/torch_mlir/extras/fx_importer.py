# Copyright 2023 Advanced Micro Devices, Inc
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import logging
import operator
import re
from types import NoneType, BuiltinMethodType, BuiltinFunctionType
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import weakref

import numpy as np

import torch
import torch.export
import torch.fx as torch_fx
from torch.fx.passes.shape_prop import TensorMetadata

from torch import (
    dtype as TorchDtype,
    FunctionSchema,
)

from torch._ops import (
    OpOverload as TorchOpOverload,
)

from torch._subclasses import (
    FakeTensor as TorchFakeTensor,
)

from torch.fx import (
    Graph,
    GraphModule,
)

from torch.fx.node import (
    Argument as NodeArgument,
)

from ..ir import (
    Attribute,
    Block,
    Context,
    DenseResourceElementsAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    RankedTensorType,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
    Type as IrType,
    Value,
)

from ..dialects import (
    func as func_dialect,
)

__all__ = [
    "FxImporter",
]

# An external callback that, given a Python value and a GraphNodeImporter, may choose
# to materialize IR to load the value as a vtensor. If it returns None, then default
# literal resolution proceeds.
LiteralResolverCallback = Callable[[Any, "GraphNodeImporter"], Optional[Value]]

REQUIRED_DIALCTS = [
    "builtin",
    "func",
    "torch",
]

TORCH_DTYPE_TO_MLIR_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "ui8",
    torch.int8: "si8",
    torch.int16: "si16",
    torch.int32: "si32",
    torch.int64: "si64",
    torch.bool: "i1",
    torch.qint8: "!torch.qint8",
    torch.quint8: "!torch.quint8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
}

TORCH_DTYPE_TO_MLIR_TYPE: Dict[torch.dtype, Callable[[], IrType]] = {
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.uint8: lambda: IntegerType.get_unsigned(8),
    torch.int8: lambda: IntegerType.get_signed(8),
    torch.int16: lambda: IntegerType.get_signed(16),
    torch.int32: lambda: IntegerType.get_signed(32),
    torch.int64: lambda: IntegerType.get_signed(64),
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.qint8: lambda: IntegerType.get_signed(8),
    torch.quint8: lambda: IntegerType.get_unsigned(8),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
    torch.complex128: lambda: ComplexType.get(F64Type.get()),
}

TORCH_DTYPE_TO_NPY_TYPE = {
    # torch.qint8: None, # no equivalent np datatype
    # torch.quint8: None,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    # torch.bf16: None, there's no equivalent np datatype so this isn't supported right now
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
    # torch.complex32: None, # no equivalent precision for numpy
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

TORCH_DTYPE_TO_INT = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    # torch.complex_half 8
    torch.complex32: 9,
    torch.complex64: 10,
    torch.bool: 11,
    # torch.qint8: 12, # quantized dtypes are not supported in all backends, currently we do not support them
    # torch.quint8: 13,
    # torch.qint32 14
    torch.bfloat16: 15,
}

TORCH_MEMORY_FORMAT_TO_INT = {
    torch.contiguous_format: 0,
    torch.preserve_format: 1,
    torch.channels_last: 2,
    torch.channels_last_3d: 3,
}

TORCH_LAYOUT_TO_INT = {
    torch.strided: 0,
    torch.sparse_coo: 1,
    torch.sparse_csr: 2,
    torch.sparse_csc: 3,
    torch.sparse_bsr: 4,
    torch.sparse_bsc: 5,
}

PY_BUILTIN_TO_TORCH_OP = {
    "truediv": torch.ops.aten.div,
    "mul": torch.ops.aten.mul,
    "add": torch.ops.aten.add,
    "sub": torch.ops.aten.sub,
    "lt": torch.ops.aten.lt,
    "le": torch.ops.aten.le,
    "ge": torch.ops.aten.ge,
    "ne": torch.ops.aten.ne,
    "gt": torch.ops.aten.gt,
}

SYMBOLIC_TORCH_OPS = {
    torch.ops.aten.sym_size,
    torch.ops.aten.sym_stride,
    torch.ops.aten.sym_numel,
}

SYMBOLIC_OP_TO_TORCH_OP = {
    (torch.ops.aten.sym_size, 1): torch.ops.aten.size.default,
    (torch.ops.aten.sym_size, 2): torch.ops.aten.size.int,
    (torch.ops.aten.sym_stride, 1): torch.ops.aten.stride.default,
    (torch.ops.aten.sym_stride, 2): torch.ops.aten.stride.int,
    (torch.ops.aten.sym_numel, 1): torch.ops.aten.numel.default,
}


"""Check whether an object in our graph is symbolic"""


def is_symbolic(obj: Any) -> bool:
    return isinstance(obj, (torch.SymInt, torch.SymFloat, torch.SymBool))


def is_builtin_function_or_method(obj: Any) -> bool:
    return isinstance(obj, (BuiltinMethodType, BuiltinFunctionType))


class FxImporter:
    """Main entry-point for importing an fx.GraphModule.

    The FxImporter is a low-level class intended for framework integrators.
    It provides several options for customization:

    * config_check: Optionally allows some per-import configuration safety
      checks to be skipped.
    * literal_resolver_callback: Callback that will be invoked when a literal,
      live torch.Tensor is encountered in the FX graph, allowing the default
      action (which is to inline the data as a DenseResourceElementsAttr) to
      be completely overriden.
    * py_attr_tracker: Weak reference tracker for live PyTorch objects used
      to unique them with respect to attributes. If not specified, there will
      be one reference tracker per import, but this can be injected to share
      the same uniqueing across imports (i.e. if building multiple functions
      into the same context or module).
    """

    __slots__ = [
        "_c",
        "_cc",
        "_literal_resolver_callback",
        "_m",
        "_m_ip",
        "_py_attr_tracker",
        "symbol_table",
    ]

    def __init__(
        self,
        *,
        module: Optional[Module] = None,
        context: Optional[Context] = None,
        config_check: bool = True,
        literal_resolver_callback: Optional[LiteralResolverCallback] = None,
        py_attr_tracker: Optional["RefTracker"] = None,
    ):
        if module is not None:
            assert context is None, "If configuring with a Module, context must be None"
            self._m = module
            self._c = self.module.context
        else:
            self._c = context if context else Context()
            self._m = Module.create(Location.unknown(self._c))
        if config_check:
            # Production code can disable this for a bit of a boost.
            self._config_check()
        self._py_attr_tracker = py_attr_tracker or RefTracker()
        self._cc = ContextCache(self._c, py_attr_tracker=self._py_attr_tracker)
        self._m_ip = InsertionPoint(self._m.body)
        self._literal_resolver_callback = literal_resolver_callback
        self.symbol_table = SymbolTable(self._m.operation)

    def _config_check(self):
        for dname in REQUIRED_DIALCTS:
            try:
                self._c.dialects[dname]
                logging.debug("Context has registered dialect '%s'", dname)
            except IndexError:
                raise RuntimeError(
                    f"The MLIR context {self._c} is missing required dialect '{dname}'"
                )

    @property
    def module(self) -> Module:
        return self._m

    @property
    def module_op(self) -> Operation:
        return self._m.operation

    def import_frozen_exported_program(self, prog: torch.export.ExportedProgram):
        """Imports a consolidated torch.export.ExportedProgram instance.

        If using the new torch.export path (vs a lower level precursor), then this is
        the recommended way to canonically use this importer.

        The ExportedProgram form differs from some of the earlier work primarily in
        how it deals with references to external tensors from "outside". In this form,
        all such references are checked to have originated from within the exported
        scope or from an @assume_constant_result wrapped function. Then they are
        transformed to graph inputs and stashed in one of two data structures on
        the ExportedProgram:
        inputs_to_buffers / buffers : For non-parameter buffers.
        inputs_to_parameters / parameters : For parameter buffers.
        The values of the mapping in inputs_to_{buffers|parameters} are in the
        state_dict. This replaces get_attr nodes that would have classically been
        present during lower level tracing.
        Historically, torch-mlir has assumed that all such external accesses are
        frozen, and this entry-point preserves this behavior, treating each distinct
        torch.Tensor encountered in such a way as a `torch.vtensor.literal` (or
        delegating to the literal_resolver_callback to make a policy decision).

        As we anticipate more nuanced treatment options in the future, we name this
        method to indicate that it is producing "frozen" modules. Additional top-level
        approaches to handling state can be introduced later as an addition.
        """
        sig = prog.graph_signature
        state_dict = prog.state_dict
        arg_replacements: dict[str, Any] = {}
        # Lift buffers.
        for input_name, state_name in sig.inputs_to_buffers.items():
            try:
                state_value = state_dict[state_name]
            except KeyError as e:
                raise AssertionError("Could not find state mapping for buffer") from e
            arg_replacements[input_name] = state_value

        # Lift parameters.
        for input_name, state_name in sig.inputs_to_parameters.items():
            try:
                state_value = state_dict[state_name]
            except KeyError as e:
                raise AssertionError(
                    "Could not find state mapping for parameter"
                ) from e
            arg_replacements[input_name] = state_value

        # Remove any lifted placeholders, replacing their uses with the state 
        # replacement value.
        g = prog.graph
        for node in g.nodes:
            if node.op == "placeholder":
                replacement = arg_replacements.get(node.name)
                if replacement is None:
                    continue
                node.replace_all_uses_with(replacement)
                g.erase_node(node)

        self.import_stateless_graph(g)

    def import_graph_module(self, gm: GraphModule):
        """Low-level import of a GraphModule assuming that it has been functionalized."""
        self.import_stateless_graph(gm.graph)

    def import_stateless_graph(self, g: Graph, func_name: str = "main"):
        """Low-level import of a functionalized, assumed stateless Graph as a func."""
        ftype, loc = self._graph_to_function_meta(g)
        # TODO: The FuncOp constructor requires a context-manager context.
        # Fix upstream and then unnest.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/138
        with loc:
            func = func_dialect.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
            )
            entry_block = Block.create_at_start(func.body, ftype.inputs)
        node_importer = GraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
            literal_resolver_callback=self._literal_resolver_callback,
        )
        node_importer.import_nodes(g.nodes)
        self.symbol_table.insert(func)

    def _graph_to_function_meta(self, g: Graph) -> Tuple[FunctionType, Location]:
        """Extracts function metadata from the Graph.

        Principally, this includes the FunctionType, but in the future,
        it should also return other annotations (input strides, etc) that
        affect compilation and should be included as arg attrs.
        """
        input_types = []
        result_types = []
        loc = None
        for node in g.nodes:
            # Assume that the first node we can get a location for is about as
            # good as it gets as an overall function location.
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                input_types.append(self._cc.node_val_to_type(node))
            elif node.op == "output":
                # An output node's args[0] is the return value. This seems to
                # always be "boxed" as a tuple, which we emit as multi-results.
                for result_node in node.args[0]:
                    if result_node is None:
                        result_types.append(
                            IrType.parse("!torch.none", context=self._c)
                        )
                    else:
                        result_types.append(self._cc.node_val_to_type(result_node))
        return (
            FunctionType.get(input_types, result_types, context=self._c),
            loc if loc else Location.unknown(self._c),
        )


class ContextCache:
    """Caches per-context lookups of various things that we ask for repeatedly."""

    __slots__ = [
        "_c",
        "_dtype_to_type",
        "_tensor_metadata_cache",
        "_py_attr_tracker",
        # Types.
        "torch_bool_type",
        "torch_float_type",
        "torch_int_type",
        "torch_none_type",
        "torch_str_type",
        "torch_device_type",
    ]

    def __init__(
        self, context: Context, *, py_attr_tracker: Optional["RefTracker"] = None
    ):
        self._c = context
        self._dtype_to_type: Dict[TorchDtype, IrType] = {}
        self._tensor_metadata_cache: Dict[Tuple[torch.Size, torch.dtype], IrType] = {}
        self._py_attr_tracker = py_attr_tracker or RefTracker()

        # Common types.
        with context:
            self.torch_bool_type = IrType.parse("!torch.bool")
            self.torch_float_type = IrType.parse("!torch.float")
            self.torch_int_type = IrType.parse("!torch.int")
            self.torch_none_type = IrType.parse("!torch.none")
            self.torch_str_type = IrType.parse("!torch.str")
            self.torch_device_type = IrType.parse("!torch.Device")

    def integer_attr(self, value: int, bits: int) -> Attribute:
        c = self._c
        return IntegerAttr.get(IntegerType.get_signless(bits, c), value)

    """Strips symbolic elements from a torch.Size object and returns shape asm"""

    def format_asm_shape(self, shape: torch.Size) -> str:
        return ",".join("?" if is_symbolic(d) else str(d) for d in list(shape))

    """Return IrType for !torch.vtensor with the given shape and dtype"""

    def get_vtensor_type(self, shape: torch.Size, dtype: torch.dtype):
        shape_asm = self.format_asm_shape(shape)
        mlir_dtype = str(self.dtype_to_type(dtype))
        return IrType.parse(
            f"!torch.vtensor<[{shape_asm}],{str(mlir_dtype)}>", context=self._c
        )

    def node_val_to_type(self, node: torch_fx.Node) -> IrType:
        try:
            tensor_meta = node.meta.get("tensor_meta")
            val = node.meta.get("val")
            if tensor_meta is not None:
                assert isinstance(tensor_meta, TensorMetadata)
                # Quantized tensor meta data is not preserved in our lowering,
                # so throw error instead of silently doing wrong thing.
                if tensor_meta.is_quantized:
                    raise NotImplementedError(
                        f"Quantized tensor meta data is not supported."
                    )
                else:
                    return self.tensor_metadata_to_type(tensor_meta)
            elif val is not None:
                # some nodes with symbolic inputs pass a 'val' attribute rather than
                # tensor_meta
                if isinstance(val, TorchFakeTensor):
                    return self.get_vtensor_type(val.size(), val.dtype)

                t = SCALAR_TYPE_TO_TORCH_MLIR_TYPE.get(type(val))
                if t is not None:
                    return IrType.parse(t, self._c)

            raise NotImplementedError(
                f"FIXME: Unsupported placeholder node (this often indicates that a necessary) "
                f"fx preprocessing pass was not run): {node.meta}"
            )
        except KeyError as e:
            raise RuntimeError(
                f"FIXME: Illegal access to torch.fx.Node.meta: {e} ({node.meta.keys()} : {node.meta})"
            )

    def tensor_metadata_to_type(self, tm: TensorMetadata) -> IrType:
        tm_shape = tuple(
            item.node if is_symbolic(item) else item for item in list(tm.shape)
        )

        key = (tm_shape, tm.dtype)
        t = self._tensor_metadata_cache.get(key)
        if t is None:
            t = self.get_vtensor_type(tm.shape, tm.dtype)
            self._tensor_metadata_cache[key] = t
        return t

    def dtype_to_type(self, dtype: TorchDtype) -> IrType:
        t = self._dtype_to_type.get(dtype)
        if t is None:
            try:
                asm = TORCH_DTYPE_TO_MLIR_TYPE_ASM[dtype]
            except IndexError:
                raise ValueError(f"Unknown conversion from {dtype} to IREE type")
            t = IrType.parse(asm, self._c)
            self._dtype_to_type[dtype] = t
        return t

    def tensor_to_vtensor_type(self, tensor: torch.Tensor) -> IrType:
        dtype_asm = str(self.dtype_to_type(tensor.dtype))
        return IrType.parse(f"!torch.vtensor<{list(tensor.size())},{dtype_asm}>")

    def get_node_location(self, node: torch_fx.Node) -> Optional[Location]:
        stack_trace = node.meta.get("stack_trace")
        if stack_trace is None:
            return None
        # Ugh.
        # TODO: Avoid needing to regex match this.
        # https://github.com/pytorch/pytorch/issues/91000
        stack_trace = node.stack_trace
        if stack_trace:
            m = re.search(r"""File "([^"]+)", line ([0-9]+),""", stack_trace)
            if m:
                filename, line = m.group(1), int(m.group(2))
                return Location.file(filename, line, col=0, context=self._c)
        return Location.unknown(context=self._c)


class GraphNodeImporter:
    """Imports graph nodes into an MLIR function.

    The caller must have already created the function.
    """

    __slots__ = [
        "_b",
        "_c",
        "_cc",
        "_literal_resolver_callback",
        "_v",
        "_multi_result_nodes",
        "fx_importer",
    ]

    def __init__(
        self,
        fx_importer: FxImporter,
        context: Context,
        context_cache: ContextCache,
        block: Block,
        *,
        literal_resolver_callback: Optional[LiteralResolverCallback] = None,
    ):
        self.fx_importer = fx_importer
        self._c = context
        self._cc = context_cache
        self._b = block
        # Map of (Node, result_index) to MLIR Value.
        self._v: Dict[Tuple[torch_fx.Node, int], Value] = {}
        # Statically multi-result nodes which we have de-tupled are noted here.
        # They will have their getitem calls short-circuited.
        self._multi_result_nodes: Set[torch_fx.Node] = set()
        self._literal_resolver_callback = literal_resolver_callback

    def import_nodes(self, nodes: Sequence[torch_fx.Node]):
        with InsertionPoint(self._b):
            loc = Location.unknown()
            num_placeholders = 0
            for node in nodes:
                op = node.op
                # Attempt to extract locations. Not everything has them,
                # so we do our best.
                new_loc = self._cc.get_node_location(node)
                if new_loc is not None:
                    loc = new_loc
                if op == "placeholder":
                    # Associate the placeholder node with corresponding block
                    # argument.
                    self._v[(node, 0)] = self._b.arguments[num_placeholders]
                    num_placeholders += 1
                elif op == "call_function":
                    target = node.target
                    if target == operator.getitem:
                        # Special case handling of getitem for when it is resolving
                        # against a function call that we know has returned multiple
                        # results. We short-circuit this case because we have modeled
                        # function calls to natively return multiple results vs tupling.
                        getitem_ref, getitem_index = node.args
                        if getitem_ref in self._multi_result_nodes:
                            try:
                                self._v[(node, 0)] = self._v[
                                    (getitem_ref, getitem_index)
                                ]
                            except IndexError:
                                raise RuntimeError(
                                    f"getitem de-aliasing failed. This likely "
                                    f"indicates a programmer error that usually "
                                    f"would have happened at runtime. Please "
                                    f"notify developers if this case happens "
                                    f"(at {loc})."
                                )
                        else:
                            raise NotImplementedError(
                                f"General getitem access to non-multi-result ops"
                            )
                    elif isinstance(target, TorchOpOverload):
                        # Dispatch to an ATen op.
                        self._import_torch_op_overload(loc, node, target)
                    elif target in SYMBOLIC_TORCH_OPS or (
                        is_symbolic(node.meta.get("val"))
                        and is_builtin_function_or_method(target)
                    ):
                        self._import_symbolic_torch_op(loc, node, target)
                    else:
                        raise NotImplementedError(
                            f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                        )
                elif op == "output":
                    # args[0] is a singleton tuple that we flatten into multiple
                    # results.
                    operands = [self._import_argument(loc, arg) for arg in node.args[0]]
                    func_dialect.ReturnOp(operands, loc=loc)

    def _promote_symbolic_scalar_int_float(self, loc, graph, param):
        temp_target = torch.ops.aten.Float.Scalar
        temp_node = torch.fx.Node(
            graph=graph,
            name=f"{str(param)}_as_float",
            op="call_function",
            target=temp_target,
            args=(param,),
            kwargs={},
            return_type=float,
        )
        temp_node.meta["val"] = torch.sym_float(param.meta["val"])
        self._import_torch_op_overload(loc, temp_node, temp_target)
        return temp_node

    def _import_symbolic_torch_op(
        self,
        loc: Location,
        node: torch_fx.Node,
        target: Union[
            torch._ops.OpOverloadPacket, BuiltinMethodType, BuiltinFunctionType
        ],
    ):
        # parse builtin operations like add, sub, mul, etc. because dynamo captures these
        # operations on symbolic arguments as regular python expressions rather than as torch ops
        if is_builtin_function_or_method(target):
            arg_types = [
                arg.meta["val"].node.pytype
                if isinstance(arg, torch.fx.Node)
                else type(arg)
                for arg in node.args
            ]
            is_int = [item == int for item in arg_types]
            if all(is_int):
                op_overload = "int"
            elif any(is_int):
                if target.__name__ in ("add", "lt", "ge", "ne", "gt"):
                    op_overload = "float_int"
                    # put float arg first, as expected in signature
                    if arg_types[1] == float:
                        node.args = (node.args[1], node.args[0])
                else:
                    # promote int argument to float - following torch-mlir convention
                    arg0, arg1 = node.args
                    if is_int[0]:
                        if isinstance(arg0, torch.fx.Node):
                            prom_arg = self._promote_symbolic_scalar_int_float(
                                loc, node.graph, arg0
                            )
                            new_args = (prom_arg, arg1)
                        else:
                            arg0 = float(arg0)
                            new_args = (arg0, arg1)
                    else:
                        if isinstance(arg1, torch.fx.Node):
                            prom_arg = self._promote_symbolic_scalar_int_float(
                                loc, node.graph, arg1
                            )
                            new_args = (arg0, prom_arg)
                        else:
                            arg1 = float(arg1)
                            new_args = (arg0, arg1)

                    node.args = new_args
                    op_overload = "float"
            else:
                op_overload = "float"

            torch_op = PY_BUILTIN_TO_TORCH_OP.get(target.__name__)
            assert (
                torch_op is not None
            ), f"Unsupported builtin function for symbolic types: {target} with args {node.args}"
            concrete_target = getattr(torch_op, op_overload)
        else:
            concrete_target = SYMBOLIC_OP_TO_TORCH_OP.get((target, len(node.args)))

        assert (
            concrete_target is not None
        ), f"Unable to parse symbolic operation: {target} with args {node.args}"
        self._import_torch_op_overload(loc, node, concrete_target)

    def _import_torch_op_overload(
        self, loc: Location, node: torch_fx.Node, target: TorchOpOverload
    ):
        # replace lift_fresh_copy with clone op
        if target == torch.ops.aten.lift_fresh_copy.default:
            node.target = target = torch.ops.aten.clone.default
            node.args = (node.args[0], None)
        elif target == torch.ops.aten.lift_fresh_copy.out:
            node.target = target = torch.ops.aten.clone.out
            node.args = (node.args[0], None, node.args[1])
        # TODO: generalize empty.memory_format in the future
        # Currently, the aten.baddbmm.default op for Unet includes multiplying an
        # empty.memory_format input with a constant, which creates NaN values
        # because empty.memory_format contains uninitialized data. Converting
        # aten.baddbmm.default -> aten.zeros.default fixes the correctness issue
        elif target == torch.ops.aten.empty.memory_format:
            if len(node.users) == 1:
                for key_node in node.users:
                    if key_node.target == torch.ops.aten.baddbmm.default:
                        node.target = target = torch.ops.aten.zeros.default

        schema = target._schema
        assert isinstance(schema, FunctionSchema)

        # Map to a `torch` dialect name.
        namespace, sep, unqualified_name = schema.name.partition("::")
        assert sep, f"Malformed Torch op name {schema.name}"
        mlir_op_name = f"torch.{namespace}.{unqualified_name}"
        if schema.overload_name != "":
            mlir_op_name += f".{schema.overload_name}"

        # Intervening to use Scalar ops due to incorrect ops from AOT-autograd with scalar arguments.
        if mlir_op_name in TENSOR_SCALAR_OP_CONVERTER and (
            isinstance(node.args[1], float) or isinstance(node.args[1], int)
        ):
            mlir_op_name = TENSOR_SCALAR_OP_CONVERTER[mlir_op_name]
            # we are dynamically changing which op is emitted here due to an issue in
            # torch dynamo where it emits the Tensor variant of ops even when processing
            # scalar arguments, therefore we retrieve the schema as well so that we
            # consume the correct typing information when subsequently importing the
            # function arguments and result types
            # i.e. the code below is basically doing `schema = torch.ops.aten.my_op.Scalar._schema`
            op_attrs = mlir_op_name.split(".")
            op_overload = getattr(torch, "ops")
            for i in range(1, len(op_attrs)):
                op_overload = getattr(op_overload, op_attrs[i])
            schema = op_overload._schema

        return_count = len(schema.returns)
        if return_count == 1:
            # Unary return directly maps a single meta["val"] and cannot be subscripted.
            # if "tensor_meta" is None, this will throw unsupported placeholder node error
            result_types = [self._cc.node_val_to_type(node)]
        elif return_count == 0:
            # Some torch ops do have 0 returns, and these are supported with ZeroResults
            # op trait. Python bindings for IR creation allow us to pass empty result_types
            # for such ops. Therefore, we pass an empty result types for these cases.
            result_types = []
        else:
            # Multi-return will unpack the meta["val"] and trigger our getitem subscripting
            # short-circuit above. Note that if we ever choose to also fully reify Python
            # level result tuples, we will need to create a tuple-boxed version of this and
            # redirect to it for generic object access.

            result_types = []
            for v in node.meta["val"]:
                result_types.append(self._cc.tensor_metadata_to_type(v))
            result_types = tuple(result_types)

            self._multi_result_nodes.add(node)
        # Unroll operands from formal parameters, args and kwargs.
        operands = []
        for i, parameter in enumerate(schema.arguments):
            if parameter.kwarg_only and parameter.name in node.kwargs:
                operands.append(
                    self._import_argument(
                        loc, node.kwargs[parameter.name], parameter.type
                    )
                )
            elif i < len(node.args):
                operands.append(
                    self._import_argument(loc, node.args[i], parameter.type)
                )
            else:
                operands.append(
                    self._import_default_value(
                        loc, parameter.default_value, parameter.type
                    )
                )

        # Support unregistered torch ops using torch.operator.
        # torch.operator is used to represent ops from registry
        # which haven't been generated by torch_ods_gen.py.
        if not self._c.is_registered_operation(mlir_op_name):
            operation = Operation.create(
                "torch.operator",
                attributes={"name": StringAttr.get(mlir_op_name)},
                results=result_types,
                operands=operands,
                loc=loc,
            )
        else:
            operation = Operation.create(
                mlir_op_name,
                results=result_types,
                operands=operands,
                loc=loc,
            )

        # Record value mapping.
        for i, value in enumerate(operation.results):
            self._v[(node, i)] = value

    def _import_argument(
        self, loc: Location, arg: NodeArgument, expected_jit_type=None
    ) -> Value:
        """Import an FX `Argument`, which must result to an MLIR `Value`."""
        if isinstance(arg, torch_fx.Node):
            # If implementing boxed support for multi-result nodes, then
            # this will need to do something more intelligent.
            if arg in self._multi_result_nodes:
                raise RuntimeError(f"Attempt to de-reference a multi-result node")

            # catch references to dynamically created constant attributes and make sure they have an origin in our module
            if arg.op == "get_attr" and (arg.target, 0) not in self._v:
                gm = arg.graph.owning_module
                assert hasattr(
                    gm, arg.target
                ), f"Attempting to retrieve attribute '{arg.target}' from module, but no such attribute exists"
                obj = getattr(gm, arg.target)
                with loc:
                    self._v[(arg, 0)] = self._import_literal(obj)

            return self._v[(arg, 0)]
        elif isinstance(arg, torch_fx.immutable_collections.immutable_list):
            return self._import_list_argument(loc, arg, expected_jit_type)
        elif isinstance(expected_jit_type, torch.TensorType) and not isinstance(
            arg, torch.Tensor
        ):
            # promote scalars to tensor types as appropriate
            return self._import_scalar_as_tensor(loc, arg)
        else:
            with loc:
                return self._import_literal(arg)

    def _import_literal(self, py_value: Any) -> Value:
        # Apply the conversion callback.
        user_callback = self._literal_resolver_callback
        if user_callback:
            user_value = user_callback(py_value, self)
            if user_value is not None:
                assert isinstance(user_value, Value)
                return user_value

        # Default conversion path.
        converter = LITERAL_CONVERTER_MAP.lookup(type(py_value))
        if converter is None:
            raise TypeError(
                f"Unsupported argument -> literal conversion for {py_value.__class__}"
            )
        return converter(py_value, self, self._cc)

    def _import_scalar_as_tensor(self, loc: Location, arg: NodeArgument) -> Value:
        tensor_arg = torch.tensor(arg)
        result_type = self._cc.get_vtensor_type(tensor_arg.size(), tensor_arg.dtype)
        with loc:
            constant_arg = LITERAL_CONVERTER_MAP.lookup(type(arg))(arg, self, self._cc)

        return Operation.create(
            name="torch.prim.NumToTensor.Scalar",
            results=[result_type],
            operands=[constant_arg],
            loc=loc,
        ).result

    def _import_list_argument(
        self, loc: Location, arg: NodeArgument, expected_jit_type
    ) -> Value:
        assert (
            isinstance(expected_jit_type, torch.ListType)
            or (
                isinstance(expected_jit_type, torch.OptionalType)
                and isinstance(expected_jit_type.getElementType(), torch.ListType)
            )
            or isinstance(expected_jit_type, NoneType)
        ), f"Unexpected jit type as list argument: {arg} of type {expected_jit_type}"

        # parse list type
        if expected_jit_type is None:
            element_type = type(arg[0])
        else:
            element_jit_type = expected_jit_type.getElementType()

            # this branch is needed to handle Optional[List[]] types
            if isinstance(element_jit_type, torch.ListType):
                element_jit_type = element_jit_type.getElementType()

            # this handles getting the inner types for List[Optional[]] types
            is_optional_type = isinstance(element_jit_type, torch.OptionalType)
            if is_optional_type:
                element_jit_type = element_jit_type.getElementType()
            element_type = TORCH_TYPE_TO_PY_TYPE[type(element_jit_type)]

        # create list operands
        list_operands = []

        for operand in arg:
            operand_type = type(operand)
            if isinstance(operand, torch.fx.Node):
                if operand in self._multi_result_nodes:
                    raise RuntimeError(f"Attempt to de-reference a multi-result node")
                val = self._v[(operand, 0)]
                val_type = str(val.type)
                assert (
                    isinstance(element_type, str) and element_type in val_type
                ) or SCALAR_TYPE_TO_TORCH_MLIR_TYPE.get(
                    element_type
                ) == val_type, f"Heterogeneous lists are not supported: expected {element_type}, got {val_type}"
            else:
                assert (is_optional_type and operand_type is NoneType) or (
                    element_type == operand_type
                ), f"Heterogeneous lists are not supported: expected {element_type}, got {operand_type}"

                operand_jit_type = (
                    torch.NoneType if operand_type is NoneType else element_jit_type
                )
                val = self._import_default_value(loc, operand, operand_jit_type)

            list_operands.append(val)

        # construct list op
        if is_optional_type:
            list_type = PY_TYPE_TO_TORCH_OPTIONAL_LIST_TYPE[element_type]
        else:
            list_type = PY_TYPE_TO_TORCH_LIST_TYPE[element_type]

        result_type = IrType.parse(list_type, context=self._c)
        operation = Operation.create(
            "torch.prim.ListConstruct",
            results=[result_type],
            operands=list_operands,
            loc=loc,
        )

        return operation.result

    def _import_default_value(self, loc: Location, arg, expected_jit_type) -> Value:
        """Imports a defaulted value for a known function schema."""
        if isinstance(arg, list):
            return self._import_list_argument(loc, arg, expected_jit_type)

        # The LITERAL_CONVERTER_MAP maps each arg to its respective constant
        # of the expected jit IR type (types like torch.dtype will form a chain of
        # maps to get to constant of expected_jit_type).
        cvt = LITERAL_CONVERTER_MAP.lookup(type(arg))
        if cvt is None:
            raise RuntimeError(f"Unhandled default value ({arg.__class__}): {arg})")
        with loc:
            return cvt(arg, self, self._cc)


def _make_constant_op(
    op_name: str, value_attr: Attribute, result_type: Optional[IrType] = None
) -> Operation:
    return Operation.create(
        op_name,
        results=[result_type if result_type else value_attr.type],
        attributes={"value": value_attr},
    )


def create_mlir_tensor_type(tensor: torch.Tensor) -> IrType:
    try:
        dtype = tensor.dtype
        element_type = TORCH_DTYPE_TO_MLIR_TYPE[dtype]()
        tensor_type = RankedTensorType.get(tuple(tensor.size()), element_type)
        return tensor_type
    except KeyError:
        raise TypeError(f"Could not map Torch dtype {dtype} to an IREE type")


def _make_vtensor_literal_op(
    tensor: torch.Tensor, vtensor_type: IrType, py_attr_tracker: "RefTracker"
) -> Operation:
    mapping = py_attr_tracker.track(tensor)
    if mapping.is_empty:
        # Resolve the attribute.
        npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
        assert (
            npy_dtype is not None
        ), f"Can not create literal tensor for unsupported datatype: {tensor.dtype}"
        # We need a raw buffer of data in order to create an ElementsAttr for the invocation of torch.vtensor.literal,
        # but torch.Tensor does not fulfill the python buffer/array interface hence we must convert to a numpy array to get
        # a raw buffer of our data. We can't call torch.Tensor.numpy() directly because this internally forces a call to
        # detach() which throws an error as we are operating in a FakeTensorMode, hence the simplest way to get this raw
        # buffer is via the indirection: Tensor -> list -> numpy array. This allows us to create a vtensor literal as
        # desired, but also limits which data types we can support in this function (see TORCH_DTYPE_TO_NPY_TYPE above)
        np_tensor = np.array(tensor.tolist()).astype(npy_dtype)
        bytes_view = memoryview(np_tensor)
        tensor_type = create_mlir_tensor_type(tensor)
        shape_desc = "_".join([str(d) for d in tensor.shape])
        blob_name = f"torch_tensor_{shape_desc}_{str(tensor.dtype)}"
        elements_attr = DenseResourceElementsAttr.get_from_buffer(
            bytes_view,
            blob_name,
            tensor_type,
        )
        mapping.value = elements_attr
    else:
        elements_attr = mapping.value
    return Operation.create(
        name="torch.vtensor.literal",
        results=[vtensor_type],
        attributes={"value": elements_attr},
    )


################################################################################
# TypeSubclassMapping
################################################################################


class TypeSubclassMap:
    """Mapping of super-types to values.

    Maintains a cache of actual types seen and uses that instead of a linear
    scan.
    """

    __slots__ = [
        "_cache",
        "_mapping",
    ]

    def __init__(self):
        # The linear list of converters.
        self._mapping: List[Tuple[type, Any]] = []
        # When there is a hit on the linear mapping, memoize it here.
        self._cache: Dict[type, Any] = {}

    def map(self, t: type, value: Any):
        self._mapping.append((t, value))
        self._cache[t] = value

    def lookup(self, t: type) -> Any:
        try:
            return self._cache[t]
        except KeyError:
            pass
        for t_super, value in self._mapping:
            if issubclass(t, t_super):
                self._cache[t] = value
                return value
        else:
            self._cache[t] = None
            return None


###############################################################################
# Reference mapping
###############################################################################


# Opaque value to indicate something is empty. Used in cases where 'None'
# may have a different meaning.
class EmptyType:
    ...


Empty = EmptyType()


class RefMapping:
    __slots__ = [
        "_referrent",
        "value",
    ]

    def __init__(self, referrent: Any):
        if referrent is not Empty:
            self._referrent = weakref.ref(referrent)
        self.value = Empty

    @property
    def is_empty(self):
        return self.value is Empty

    def __repr__(self):
        return (
            f"<RefMapping {id(self._referrent) if self._referrent is not Empty else 'empty'} -> "
            f"{self.value if self.value is not Empty else 'empty'}>"
        )


class RefTracker:
    """Tracks live references from Python values to symbolic associations."""

    def __init__(self):
        self._refs: Dict[int, RefMapping] = {}

    def track(self, referrent: Any) -> RefMapping:
        ref_id = id(referrent)
        existing = self._refs.get(ref_id)
        if existing:
            return existing
        info = RefMapping(referrent)
        if referrent is not Empty:
            weakref.finalize(referrent, self._ref_finalizer, ref_id)
        self._refs[ref_id] = info
        return info

    def _ref_finalizer(self, ref_id: int):
        del self._refs[ref_id]


################################################################################
# Mappings
################################################################################

LITERAL_CONVERTER_MAP = TypeSubclassMap()
LITERAL_CONVERTER_MAP.map(
    NoneType,
    lambda arg, gni, cc: Operation.create(
        "torch.constant.none", results=[cc.torch_none_type]
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    bool,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.bool", cc.integer_attr(arg, 1), cc.torch_bool_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    int,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.int", cc.integer_attr(arg, 64), cc.torch_int_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    float,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.float", FloatAttr.get_f64(arg), cc.torch_float_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    str,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.str", StringAttr.get(arg), cc.torch_str_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    torch.Tensor,
    lambda arg, gni, cc: _make_vtensor_literal_op(
        arg, cc.tensor_to_vtensor_type(arg), cc._py_attr_tracker
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    torch.device,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.device", StringAttr.get(str(arg)), cc.torch_device_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    torch.dtype,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_DTYPE_TO_INT[arg], gni, cc
    ),
)
LITERAL_CONVERTER_MAP.map(
    torch.layout,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_LAYOUT_TO_INT[arg], gni, cc
    ),
)
LITERAL_CONVERTER_MAP.map(
    torch.memory_format,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_MEMORY_FORMAT_TO_INT[arg], gni, cc
    ),
)

TORCH_TYPE_TO_PY_TYPE = {
    torch.IntType: int,
    torch.FloatType: float,
    torch.StringType: str,
    torch.BoolType: bool,
    torch.TensorType: "vtensor",
}

PY_TYPE_TO_TORCH_LIST_TYPE = {
    int: "!torch.list<int>",
    float: "!torch.list<float>",
    str: "!torch.list<str>",
    bool: "!torch.list<bool>",
    "tensor": "!torch.list<tensor>",
    "vtensor": "!torch.list<vtensor>",
}

PY_TYPE_TO_TORCH_OPTIONAL_LIST_TYPE = {
    int: "!torch.list<optional<int>>",
    float: "!torch.list<optional<float>>",
    str: "!torch.list<optional<str>>",
    bool: "!torch.list<optional<bool>>",
    "tensor": "!torch.list<optional<tensor>>",
    "vtensor": "!torch.list<optional<vtensor>>",
}

SCALAR_TYPE_TO_TORCH_MLIR_TYPE = {
    torch.SymInt: "!torch.int",
    torch.SymFloat: "!torch.float",
    torch.SymBool: "!torch.bool",
    int: "!torch.int",
    float: "!torch.float",
    str: "!torch.str",
    bool: "!torch.bool",
    NoneType: "!torch.none",
}


# AOT-autograd sometimes falsely emit tensor version op with scalar arguments.
# We may remove this dictionary, if we fix such behavior in the backend.
TENSOR_SCALAR_OP_CONVERTER = {
    "torch.aten.mul.Tensor": "torch.aten.mul.Scalar",
    "torch.aten.div.Tensor": "torch.aten.div.Scalar",
    "torch.aten.add.Tensor": "torch.aten.add.Scalar",
    "torch.aten.sub.Tensor": "torch.aten.sub.Scalar",
    "torch.aten.floor_divide": "torch.aten.floor_divide.Scalar",
}
