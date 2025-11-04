# Copyright 2023 Advanced Micro Devices, Inc
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

try:
    from types import NoneType
except ImportError:
    # python less than 3.10 doesn't have NoneType
    NoneType = type(None)

import logging
import operator
import re
import sympy
import math
from dataclasses import dataclass
from types import BuiltinMethodType, BuiltinFunctionType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
    Iterable,
)
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
    HigherOrderOperator,
)

from torch._subclasses import (
    FakeTensor as TorchFakeTensor,
)

from torch.fx import (
    Graph,
    GraphModule,
    Node,
)

try:
    from torch.export.graph_signature import InputSpec as TypingInputSpec
except ModuleNotFoundError:
    # PyTorch prior to 2.3 is missing certain things we use in typing
    # signatures. Just make them be Any.
    if not TYPE_CHECKING:
        TypingInputSpec = Any
    else:
        raise

try:
    import ml_dtypes
except ModuleNotFoundError:
    # The third-party ml_dtypes package provides some optional
    # low precision data-types. If used in this file, it is
    # conditional.
    ml_dtypes = None

try:
    from torch.utils._sympy.numbers import int_oo, IntInfinity, NegativeIntInfinity
except ModuleNotFoundError:
    # This commit on PyTorch repo introduced IntInfinity and NegativeIntInfinity:
    # https://github.com/pytorch/pytorch/commit/2229884102ac95c9dda0aeadbded1b04295d892e
    # Required module may not be present in the stable version of PyTorch.
    int_oo = None
    IntInfinity = None
    NegativeIntInfinity = None

from torch.fx.node import (
    Argument as NodeArgument,
)

from ..ir import (
    AffineAddExpr,
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineModExpr,
    AffineMulExpr,
    AffineSymbolExpr,
    Attribute,
    Block,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    FlatSymbolRefAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E5M2FNUZType,
    Float8E4M3FNUZType,
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
    UnitAttr,
    Value,
)

from ..dialects import (
    func as func_dialect,
)


__all__ = [
    "FxImporter",
]

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
# Type entries added only in torch with higher version
OPTIONAL_TORCH_DTYPE_TO_MLIR_TYPE_ASM = {
    "float8_e5m2": "f8E5M2",
    "float8_e4m3fn": "f8E4M3FN",
    "float8_e5m2fnuz": "f8E5M2FNUZ",
    "float8_e4m3fnuz": "f8E4M3FNUZ",
}
for dtype_str, dtype_asm in OPTIONAL_TORCH_DTYPE_TO_MLIR_TYPE_ASM.items():
    if hasattr(torch, dtype_str):
        TORCH_DTYPE_TO_MLIR_TYPE_ASM[getattr(torch, dtype_str)] = dtype_asm

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
# Type entries added only in torch with higher version
OPTIONAL_TORCH_DTYPE_TO_MLIR_TYPE = {
    "float8_e5m2": lambda: Float8E5M2Type.get(),
    "float8_e4m3fn": lambda: Float8E4M3FNType.get(),
    "float8_e5m2fnuz": lambda: Float8E5M2FNUZType.get(),
    "float8_e4m3fnuz": lambda: Float8E4M3FNUZType.get(),
}
for dtype_str, mlir_type in OPTIONAL_TORCH_DTYPE_TO_MLIR_TYPE.items():
    if hasattr(torch, dtype_str):
        TORCH_DTYPE_TO_MLIR_TYPE[getattr(torch, dtype_str)] = mlir_type

TORCH_DTYPE_TO_NPY_TYPE = {
    # torch.qint8: None, # no equivalent np datatype
    # torch.quint8: None,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
    # torch.complex32: None, # no equivalent precision for numpy
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}
if ml_dtypes is not None:
    TORCH_DTYPE_TO_NPY_TYPE[torch.bfloat16] = ml_dtypes.bfloat16
    TORCH_DTYPE_TO_NPY_TYPE[torch.float8_e5m2] = ml_dtypes.float8_e5m2
    TORCH_DTYPE_TO_NPY_TYPE[torch.float8_e4m3fn] = ml_dtypes.float8_e4m3fn
    TORCH_DTYPE_TO_NPY_TYPE[torch.float8_e5m2fnuz] = ml_dtypes.float8_e5m2fnuz
    TORCH_DTYPE_TO_NPY_TYPE[torch.float8_e4m3fnuz] = ml_dtypes.float8_e4m3fnuz

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
# Type entries added only in torch with higher version
OPTIONAL_TORCH_DTYPE_TO_INT = {
    "float8_e5m2": 23,
    "float8_e4m3fn": 24,
    "float8_e5m2fnuz": 25,
    "float8_e4m3fnuz": 26,
}
for dtype_str, dtype_int in OPTIONAL_TORCH_DTYPE_TO_INT.items():
    if hasattr(torch, dtype_str):
        TORCH_DTYPE_TO_INT[getattr(torch, dtype_str)] = dtype_int

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
    "mod": torch.ops.aten.fmod,
    "eq": torch.ops.aten.eq,
    "floordiv": torch.ops.aten.floordiv,
}

# torch with cuda has a __version__ that looks like  "2.1.0+cu113",
# so split by + and 0 index will always give the base version
_IS_TORCH_2_1_OR_EARLIER = torch.__version__.split("+")[0] <= "2.1.0"

# The following are maps from symbolic ops to their non symbolic equivalents.
# In <=2.1.0, imported fx graphs come with a type inspecific torch.ops.aten.sym_size
# We identify it using the number of args in the node, 1 being default, 2 being int
# In the mapping below (torch.aten.sym_size, 2) indicates len(args)=2 therefore
# map to torch.aten.size.int.
# Thankfully, newer versions provide a specific torch.ops.aten.sym_size.<type>.
# Once we drop support for <2.1.0, we can get rid of the SYMBOLIC_TORCH_OPS
# set and just check key existence in SYMBOLIC_OP_TO_TORCH_OP

if _IS_TORCH_2_1_OR_EARLIER:
    SYMBOLIC_OP_TO_TORCH_OP = {
        (torch.ops.aten.sym_size, 1): torch.ops.aten.size.default,
        (torch.ops.aten.sym_size, 2): torch.ops.aten.size.int,
        (torch.ops.aten.sym_stride, 1): torch.ops.aten.stride.default,
        (torch.ops.aten.sym_stride, 2): torch.ops.aten.stride.int,
        (torch.ops.aten.sym_numel, 1): torch.ops.aten.numel.default,
    }

    SYMBOLIC_TORCH_OPS = {key[0] for key in SYMBOLIC_OP_TO_TORCH_OP}
else:
    SYMBOLIC_OP_TO_TORCH_OP = {
        torch.ops.aten.sym_size.default: torch.ops.aten.size.default,
        torch.ops.aten.sym_size.int: torch.ops.aten.size.int,
        torch.ops.aten.sym_stride.default: torch.ops.aten.stride.default,
        torch.ops.aten.sym_stride.int: torch.ops.aten.stride.int,
        torch.ops.aten.sym_numel.default: torch.ops.aten.numel.default,
    }

    SYMBOLIC_TORCH_OPS = {key for key in SYMBOLIC_OP_TO_TORCH_OP}


@dataclass
class RangeConstraint:
    min_val: int
    max_val: int


def sympy_expr_to_semi_affine_expr(
    expr: sympy.Expr, symbols_map: Dict[str, AffineSymbolExpr]
) -> AffineExpr:
    """Translate sympy expressions to MLIR (semi-)affine expressions.

    Recursively traverse the sympy expr AST and build the affine expr.
    This is not a perfect translation. Sympy expressions are much more
    expressive and not as constrained as affine (linear) expressions are.
    However, for the most part, we don't need to support all of sympy.
    PyTorch only uses a subset of sympy for capturing and expressing
    symbolic shapes, and among what's supported, we expect the semi-affine
    expressions (https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps)
    to be sufficient.
    """
    if isinstance(expr, sympy.Symbol):
        return symbols_map[str(expr)]
    elif isinstance(expr, (int, sympy.Integer)):
        return AffineConstantExpr.get(expr)
    # This handles both add (`s0 + c`) and subtract (`s0 - c`).
    # The expression is `sympy.Add` in both cases but with args
    # (s0, c) in first case and (s0, -c) in the second case.
    elif isinstance(expr, sympy.Add):
        affine_expr = AffineConstantExpr.get(0)
        for arg in expr.args:
            affine_expr = AffineAddExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(arg, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Mul):
        affine_expr = AffineConstantExpr.get(1)
        for arg in expr.args:
            affine_expr = AffineMulExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(arg, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Pow):
        base, exp = expr.args
        # Only integer exponent is supported
        # So, s1 ** s0 isn't allowed.
        assert isinstance(exp, (int, sympy.Integer))
        assert exp > 0, "Only positive exponents supported in sympy.Pow"
        affine_expr = AffineConstantExpr.get(1)
        for _ in range(exp):
            affine_expr = AffineMulExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(base, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Mod):
        dividend, divisor = expr.args
        return AffineModExpr.get(
            sympy_expr_to_semi_affine_expr(dividend, symbols_map),
            sympy_expr_to_semi_affine_expr(divisor, symbols_map),
        )
    else:
        raise NotImplementedError(
            f"Translation of sympy.Expr of type {type(expr)} not implemented yet."
        )


def sparsity_encoding(t: torch.Tensor) -> str:
    """Returns sparse tensor encoding for the given tensor as string."""

    # Sparse tensors have the form
    #   [ <batch_dimensions> , <sparse_dimensions>, <dense_dimensions> ]
    # which map directly to MLIR types.
    dim, batch_dim, sparse_dim, dense_dim = (
        t.ndim,
        t.ndim - t.sparse_dim() - t.dense_dim(),
        t.sparse_dim(),
        t.dense_dim(),
    )
    dims = ",".join(f"d{d}" for d in range(dim))

    if t.layout is torch.sparse_coo:
        assert sparse_dim >= 2
        trail_dim = batch_dim + sparse_dim - 1
        coords = ",".join(
            f"d{d}:singleton(nonunique,soa)" for d in range(batch_dim + 1, trail_dim)
        )
        sep = "," if sparse_dim > 2 else ""
        lvls = f"d{batch_dim}:compressed(nonunique),{coords}{sep}d{trail_dim}:singleton(soa)"
        idx_dtype = t._indices().dtype  # supports uncoalesced COO tensors
    elif t.layout is torch.sparse_csr:
        assert sparse_dim == 2
        lvls = f"d{batch_dim}:dense,d{batch_dim+1}:compressed"
        idx_dtype = t.col_indices().dtype
    elif t.layout is torch.sparse_csc:
        assert sparse_dim == 2
        lvls = f"d{batch_dim+1}:dense,d{batch_dim}:compressed"
        idx_dtype = t.row_indices().dtype
    else:
        assert sparse_dim == 2
        blocksize = t.values().shape[batch_dim + 1 : batch_dim + 3]
        if t.layout is torch.sparse_bsr:
            i, j = batch_dim, batch_dim + 1
            idx_dtype = t.col_indices().dtype
        else:
            assert t.layout is torch.sparse_bsc
            j, i = batch_dim, batch_dim + 1
            idx_dtype = t.row_indices().dtype
        m, n = blocksize
        lvls = (
            f"d{i} floordiv {m}:dense,d{j} floordiv {n}:compressed,"
            f"d{i} mod {m}:dense,d{j} mod {n}:dense"
        )

    if batch_dim > 0:
        batch = ",".join(f"d{d}:batch" for d in range(batch_dim))
        lvls = f"{batch},{lvls}"

    if dense_dim > 0:
        dense = ",".join(f"d{d}:dense" for d in range(batch_dim + sparse_dim, dim))
        lvls = f"{lvls},{dense}"

    posw = crdw = torch.iinfo(idx_dtype).bits
    return f"#sparse_tensor.encoding<{{map=({dims})->({lvls}),posWidth={posw},crdWidth={crdw}}}>"


def is_symbolic(obj: Any) -> bool:
    """Check whether an object in our graph is symbolic"""
    return isinstance(obj, (torch.SymInt, torch.SymFloat, torch.SymBool))


def is_builtin_function_or_method(obj: Any) -> bool:
    return isinstance(obj, (BuiltinMethodType, BuiltinFunctionType))


# TODO: switch back to `slots=True` when py3.9 support is dropped
@dataclass(frozen=True)
class InputInfo:
    """Provides additional metadata when resolving inputs."""

    program: torch.export.ExportedProgram
    input_spec: TypingInputSpec
    node: Node
    ir_type: IrType
    mutable_producer_node_name: Optional[str] = None
    store_producer_node: Optional[str] = None


class FxImporterHooks:
    """Hooks to control the behavior of the FxImporter."""

    def prepare_module(self, module_op: Operation):
        """Performs any needed preparation work on the module."""
        ...

    def resolve_literal(
        self, gni: "GraphNodeImporter", literal: Any, info: Optional[InputInfo]
    ) -> Optional[Value]:
        """User overridable hook to resolve a literal value."""
        return None

    def resolve_input(
        self, gni: "GraphNodeImporter", value: Any, info: InputInfo
    ) -> Optional[Value]:
        """Resolves a Parameter or Buffer input to an IR value.

        If the 'mutable_producer_node_name' option is set, then the result must
        be a `!torch.tensor`.
        Otherwise, it must be an immutable `!torch.vtensor`. If this constraint cannot
        be met, the implementation must either error or return None to delegate to
        the default.
        """
        return None

    def store_produced_value(
        self,
        gni: "GraphNodeImporter",
        py_value: Any,
        produced_ir_value: Any,
        info: InputInfo,
    ):
        """Given a load/store semantic mutatation, issues the store.

        This style is used for buffer and parameter updates, which are assumed to be
        non-SSA updates that are otherwise in the value-tensor domain.
        """
        raise NotImplementedError(
            f"Store of a mutation to {info} is not supported (from {produced_ir_value})"
        )


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
        "_m",
        "_m_ip",
        "_py_attr_tracker",
        "_hooks",
        "symbol_table",
        "_graph_module_to_func_name",
        "_func_name_counter",
    ]

    def __init__(
        self,
        *,
        module: Optional[Module] = None,
        context: Optional[Context] = None,
        config_check: bool = True,
        py_attr_tracker: Optional["RefTracker"] = None,
        hooks: Optional[FxImporterHooks] = None,
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
        self._hooks = hooks or FxImporterHooks()
        self.symbol_table = SymbolTable(self._m.operation)
        self._hooks.prepare_module(self._m.operation)
        # Used specifically in HOPs to map module IDs to function names
        self._graph_module_to_func_name: Dict[int, str] = {}
        # Handles collision of function names in the same module
        self._func_name_counter: int = 0

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

    def import_program(
        self,
        prog: torch.export.ExportedProgram,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Imports an ExportedProgram according to our chosen canonical representation.

        This mechanism is the fully general solution for handling an ExportedProgram
        and should eventually supercede all others. However, it depends on the
        PyTorch 2.3 release to function properly (specifically, this patch
        made ExportedProgram minimally correct for mutation:
        https://github.com/pytorch/pytorch/pull/118969).

        For stateless programs, the result of this import is a normal function
        defined for immutable `!torch.vtensors`.

        However, if the program mutates its inputs or buffers, then it will be imported
        with those parameters as `!torch.tensor` and appropriate copies and overwrites
        will be done on the inside. Note that the function is still mostly stateless,
        but with `torch.copy.to_vtensor` and `torch.overwrite.tensor.contents`
        ops at the earliest consumer or latest producer to update an argument or
        buffer.

        It is recommended that integrators subclass and override the `resolve_literal`
        method to control access to mutable buffers and parameters. Without that, the
        default policy is to capture them as frozen values.
        """
        # Create lookaside table of placeholders/outputs.
        placeholder_nodes: Dict[str, Node] = {}
        all_producer_nodes: Dict[str, Node] = {}
        loc: Optional[Location] = None
        for node in prog.graph.nodes:
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                placeholder_nodes[node.name] = node
                all_producer_nodes[node.name] = node
            elif node.op == "call_function":
                all_producer_nodes[node.name] = node
        if loc is None:
            loc = Location.unknown(self._c)

        # This API is fast evolving. We keep these imports local for now so that we
        # can disable this entire function if needed.
        from torch.export.graph_signature import (
            InputKind,
            OutputKind,
            TensorArgument,
            SymIntArgument,
            ConstantArgument,
        )

        sig = prog.graph_signature

        # Populate symbolic guards for dynamic shapes (if any)
        if import_symbolic_shape_expressions:
            self._cc.set_symbolic_guards(prog)

        # Invert the (producer, node_name) maps for mutated user inputs and mutated
        # buffers. This is because we hit-detect based on the input node name.
        mutated_user_inputs = {
            node_name: producer
            for producer, node_name in sig.user_inputs_to_mutate.items()
        }

        # Additional bindings that we need to set up after the function is created.
        mutable_buffer_target_producers: Dict[str, str] = {}
        constant_tensors: Dict[Node, torch.Tensor] = {}
        parameter_bindings: Dict[Node, Tuple[Any, InputInfo]] = {}
        buffer_bindings: Dict[Node, Tuple[Any, InputInfo]] = {}
        constant_output_values: Dict[int, Any] = {}
        constant_input_values: Dict[Node, Any] = {}

        # Derive user outputs that we preserve. These will be nodes of the
        # producer for the output.
        user_outputs: List[Optional[Node]] = []
        user_output_types: List[IrType] = []
        for i, output_spec in enumerate(sig.output_specs):
            kind = output_spec.kind
            arg = output_spec.arg
            if kind == OutputKind.USER_OUTPUT:
                if not isinstance(
                    arg, (TensorArgument, SymIntArgument, ConstantArgument)
                ):
                    raise NotImplementedError(
                        f"OutputKind.USER_OUTPUT for {type(arg)}: {arg}"
                    )
                if isinstance(arg, (TensorArgument, SymIntArgument)):
                    output_producer_node = all_producer_nodes[arg.name]
                    user_outputs.append(output_producer_node)
                    user_output_types.append(
                        self._cc.node_val_to_type(output_producer_node)
                    )
                elif isinstance(arg, ConstantArgument):
                    # Constant Outputs don't have a node so we will only store their values
                    constant_output_values[i] = arg.value
                    # Placeholder for constant outputs in the node list
                    user_outputs.append(None)
                    user_output_types.append(self._cc.value_info_to_type(arg.value))
            elif kind == OutputKind.BUFFER_MUTATION and isinstance(arg, TensorArgument):
                mutable_buffer_target_producers[output_spec.target] = arg.name

        # Derive user inputs. These will be op=='placeholder' nodes.
        user_inputs: List[Node] = []
        user_input_types: List[IrType] = []
        for input_spec in sig.input_specs:
            arg = input_spec.arg
            if input_spec.kind == InputKind.USER_INPUT:
                # Set up user input.
                if not isinstance(
                    arg, (TensorArgument, SymIntArgument, ConstantArgument)
                ):
                    raise NotImplementedError(
                        f"InputKind.USER_INPUT for {type(arg)}: {arg}"
                    )
                placeholder_node = placeholder_nodes[arg.name]
                if isinstance(arg, (TensorArgument, SymIntArgument)):
                    mutable = placeholder_node.name in mutated_user_inputs
                    user_inputs.append(placeholder_node)
                    user_input_types.append(
                        self._cc.node_val_to_type(placeholder_node, mutable=mutable)
                    )
                elif isinstance(arg, ConstantArgument):
                    # Constant argument will be handled separately, they are not mutable and do not need function parameters
                    constant_input_values[placeholder_node] = arg.value
            elif input_spec.kind == InputKind.CONSTANT_TENSOR and isinstance(
                arg, TensorArgument
            ):
                # Remember constant tensor binding.
                constant_tensors[placeholder_nodes[arg.name]] = prog.constants[
                    input_spec.target
                ]
            elif input_spec.kind == InputKind.PARAMETER and isinstance(
                arg, TensorArgument
            ):
                # Remember parameter binding.
                value = prog.state_dict.get(input_spec.target)
                assert (
                    not input_spec.persistent or value is not None
                ), "Expected state_dict value for persistent value"
                node = placeholder_nodes[arg.name]
                node_ir_type = self._cc.node_val_to_type(node, mutable=False)
                parameter_bindings[node] = (
                    value,
                    InputInfo(
                        prog,
                        input_spec,
                        node=node,
                        ir_type=node_ir_type,
                        mutable_producer_node_name=None,
                    ),
                )
            elif input_spec.kind == InputKind.BUFFER and isinstance(
                arg, TensorArgument
            ):
                # Remember buffer binding. Unlike user input mutations, buffers
                # are assumed to be represented with load/store semantics based
                # on a symbolic or other non-SSA association. As such, they
                # are not modeled with mutable IR but will trigger an output
                # store hook when the final value is produced.
                if input_spec.persistent:
                    value = prog.state_dict.get(input_spec.target)
                    assert (
                        value is not None
                    ), "Expected state_dict value for persistent buffer"
                else:
                    value = prog.constants.get(input_spec.target)
                    assert (
                        value is not None
                    ), "Expected constants value for non-persistent buffer"

                node = placeholder_nodes[arg.name]
                mutable_producer_node_name = mutable_buffer_target_producers.get(
                    input_spec.target
                )
                node_ir_type = self._cc.node_val_to_type(node, mutable=False)
                buffer_bindings[node] = (
                    value,
                    InputInfo(
                        prog,
                        input_spec,
                        node=node,
                        ir_type=node_ir_type,
                        store_producer_node=mutable_producer_node_name,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"InputSpec not of a known kind: {input_spec}"
                )

        ftype = FunctionType.get(user_input_types, user_output_types, context=self._c)

        # Create the function.
        with loc:
            func_op = func_dialect.FuncOp(
                func_name, ftype, ip=self._m_ip, visibility=func_visibility
            )
            # Programs imported from FX have strong guarantees. Setting this attribute
            # causes various lowerings to be able to emit more efficient code or
            # handle more cases. See isAssumingStrictSymbolicShapes().
            func_op.attributes["torch.assume_strict_symbolic_shapes"] = UnitAttr.get()
            entry_block = Block.create_at_start(func_op.body, ftype.inputs)

        node_importer = GraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )

        # Bind constants to IR values.
        for constant_node, constant_tensor in constant_tensors.items():
            node_importer.import_constant(loc, constant_node, constant_tensor)

        for constant_node, constant_value in constant_input_values.items():
            node_importer.import_constant(loc, constant_node, constant_value)

        # Bind user inputs to IR values.
        for user_input_node, block_arg_value in zip(user_inputs, entry_block.arguments):
            if user_input_node.name in mutated_user_inputs:
                # Materialize
                node_importer.import_mutable_to_vtensor(
                    loc,
                    user_input_node,
                    block_arg_value,
                    mutated_user_inputs[user_input_node.name],
                )
            else:
                # Normal value tensor binding.
                node_importer.bind_node_value(user_input_node, block_arg_value)

        # Lazy bind buffer and parameter inputs.
        for node, (parameter_value, info) in parameter_bindings.items():
            node_importer.lazy_import_parameter(loc, node, parameter_value, info)
        for node, (buffer_value, info) in buffer_bindings.items():
            node_importer.lazy_import_buffer(loc, node, buffer_value, info)

        # Import all child graph modules recursively for HOPs BEFORE importing nodes
        # This is necessary because HOP nodes need to reference these functions.
        # Even though import_stateless_graph is deprecated as an entrypoint mechanism,
        # HOP operator graphs are stateless graphs with no mutation, and it is correct
        # to import them as stateless graphs.
        self._import_all_child_modules(
            prog.graph.owning_module, func_name, import_symbolic_shape_expressions
        )

        # Import all nodes and return.
        node_importer.import_nodes(
            all_producer_nodes.values(),
            skip_placeholders_outputs=True,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

        # Call the return function that handles both nodes and constant values
        node_importer.return_node_values(loc, user_outputs, constant_output_values)

        self.symbol_table.insert(func_op)

        return func_op

    def _import_all_child_modules(
        self,
        module: GraphModule,
        parent_name: str,
        import_symbolic_shape_expressions: bool = False,
    ):
        """Import all child modules by delegating to import_graph_module.

        This is a thin wrapper that extracts the owning module and delegates to
        import_graph_module for each child.

        Note: This only imports children, not the parent module itself.
        """

        for child_name, child_module in module.named_children():
            if isinstance(child_module, GraphModule) and hasattr(child_module, "graph"):
                self.import_graph_module(
                    child_module,
                    func_name=child_name,
                    func_visibility="private",
                    import_symbolic_shape_expressions=import_symbolic_shape_expressions,
                )

    def import_frozen_program(
        self,
        prog: torch.export.ExportedProgram,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
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

        TODO: This mechanism should be eventually replaced by `import_program` with
        hooks set on the subclass to freeze parameters and buffers. However, that is
        waiting for the Torch 2.3 release cut.
        """
        sig = prog.graph_signature
        state_dict = prog.state_dict
        arg_replacements: Dict[str, Any] = {}

        # Populate symbolic guards for dynamic shapes (if any)
        if import_symbolic_shape_expressions:
            self._cc.set_symbolic_guards(prog)

        # If there is no "constants" attribute, consult the "state_dict". Otherwise, only look
        # at "constants". Relevant upstream patch: https://github.com/pytorch/pytorch/pull/118969
        if hasattr(prog, "constants"):
            constants = prog.constants
            # Lift tensor constants.
            for input_name, state_name in sig.inputs_to_lifted_tensor_constants.items():
                try:
                    state_value = constants[state_name]
                except KeyError as e:
                    raise AssertionError(
                        "Could not find state mapping for tensor constants"
                    ) from e
                arg_replacements[input_name] = state_value
        else:
            # Lift buffers.
            for input_name, state_name in sig.inputs_to_buffers.items():
                try:
                    state_value = state_dict[state_name]
                except KeyError as e:
                    raise AssertionError(
                        "Could not find state mapping for buffer"
                    ) from e
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

        # Import child modules for HOPs before importing the main graph
        # This ensures that any higher-order operations (like while_loop) can
        # reference the already-imported child module functions
        if hasattr(g, "owning_module") and g.owning_module is not None:
            self._import_all_child_modules(
                g.owning_module, func_name, import_symbolic_shape_expressions
            )

        return self.import_stateless_graph(
            g,
            func_name=func_name,
            func_visibility=func_visibility,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

    def import_graph_module(
        self,
        gm: GraphModule,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Low-level import of a GraphModule assuming that it has been functionalized.

        This method recursively imports all child GraphModules first, then imports
        the provided GraphModule itself. This ensures that any higher-order operations
        that reference child modules will find them already imported.

        TODO: This mechanism is deprecated by the `import_program` entry-point and
        it should be removed when no longer required for backwards compatibility.
        """
        # Store the mapping for this module itself (HOPs will need to look this up)
        module_id = id(gm)
        if module_id not in self._graph_module_to_func_name:
            # Ensure the func_name is unique
            final_func_name = func_name
            if func_name in self._graph_module_to_func_name.values():
                final_func_name = f"{func_name}_{self._func_name_counter}"
                self._func_name_counter += 1
            self._graph_module_to_func_name[module_id] = final_func_name
        else:
            # Module already imported, use existing name
            final_func_name = self._graph_module_to_func_name[module_id]

        self._import_all_child_modules(gm, func_name, import_symbolic_shape_expressions)
        # Then import this module's own graph
        return self.import_stateless_graph(
            gm.graph,
            func_name=final_func_name,
            func_visibility=func_visibility,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

    def import_stateless_graph(
        self,
        g: Graph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Low-level import of a functionalized, assumed stateless Graph as a func.

        TODO: This mechanism is deprecated by the `import_program` entry-point and
        it should be removed when no longer required for backwards compatibility.
        """
        ftype, loc = self._graph_to_function_meta(g)
        # TODO: The FuncOp constructor requires a context-manager context.
        # Fix upstream and then unnest.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/138
        with loc:
            func = func_dialect.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
                visibility=func_visibility,
            )
            entry_block = Block.create_at_start(func.body, ftype.inputs)
        node_importer = GraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )

        # Note: Child module importing is handled by import_graph_module, which is
        # the recommended entry point. This method is deprecated and should only be
        # used for stateless graphs that truly have no child modules.

        node_importer.import_nodes(
            g.nodes, import_symbolic_shape_expressions=import_symbolic_shape_expressions
        )
        self.symbol_table.insert(func)
        return func

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
                # An output node's args[0] is the return value. This is usually
                # "boxed" as a tuple, which we emit as multi-results. However,
                # for single returns it might be a single Node.
                output_arg = node.args[0]
                # Handle both single Node and tuple/list of Nodes
                result_nodes = (
                    output_arg
                    if isinstance(output_arg, (list, tuple))
                    else [output_arg]
                )

                for result_node in result_nodes:
                    if result_node is None:
                        result_types.append(
                            IrType.parse("!torch.none", context=self._c)
                        )
                    elif isinstance(result_node, torch.Tensor):
                        result_types.append(
                            self._cc.tensor_to_vtensor_type(result_node)
                        )
                    elif type(result_node) in SCALAR_TYPE_TO_TORCH_MLIR_TYPE:
                        result_types.append(
                            IrType.parse(
                                SCALAR_TYPE_TO_TORCH_MLIR_TYPE[type(result_node)],
                                self._c,
                            )
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
        "_symbolic_guards",
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
        self._tensor_metadata_cache: Dict[
            Tuple[torch.Size, torch.dtype, Optional[SparsityMeta], bool], IrType
        ] = {}
        self._symbolic_guards: Dict = {}
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

    def format_asm_shape(self, shape: torch.Size) -> str:
        """Strips symbolic elements from a torch.Size object and returns shape asm"""
        return ",".join("?" if is_symbolic(d) else str(d) for d in list(shape))

    def get_vtensor_type(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        *,
        val: Optional[torch.Tensor] = None,
        mutable: bool = False,
    ):
        """Return IrType for !torch.vtensor with the given shape and dtype"""
        stem = "torch.tensor" if mutable else "torch.vtensor"
        shape_asm = self.format_asm_shape(shape)
        mlir_dtype = str(self.dtype_to_type(dtype))
        if val is not None and val.layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            # This is a sparse tensor.
            encoding = sparsity_encoding(val)
            return IrType.parse(
                f"!{stem}<[{shape_asm}],{str(mlir_dtype)},{encoding}>",
                context=self._c,
            )
        # This is a dense tensor.
        return IrType.parse(
            f"!{stem}<[{shape_asm}],{str(mlir_dtype)}>", context=self._c
        )

    def node_val_to_type(self, node: torch_fx.Node, *, mutable: bool = False) -> IrType:
        try:
            tensor_meta = node.meta.get("tensor_meta")
            val = node.meta.get("val")
        except KeyError as e:
            raise RuntimeError(
                f"FIXME: Illegal access to torch.fx.Node.meta: {e} ({node.meta.keys()} : {node.meta})"
            )
        return self.value_info_to_type(val, tensor_meta=tensor_meta, mutable=mutable)

    def value_info_to_type(
        self,
        val,
        *,
        tensor_meta: Optional[TensorMetadata] = None,
        mutable: bool = False,
    ):
        if tensor_meta is not None:
            # separately handle when tensor_meta is a list.
            if isinstance(val, list) and all(
                isinstance(x, TorchFakeTensor) for x in val
            ):
                return IrType.parse("!torch.list<vtensor>", context=self._c)
            assert isinstance(tensor_meta, TensorMetadata)
            # Quantized tensor meta data is not preserved in our lowering,
            # so throw error instead of silently doing wrong thing.
            if tensor_meta.is_quantized:
                raise NotImplementedError(
                    f"Quantized tensor meta data is not supported."
                )
            else:
                return self.tensor_metadata_to_type(
                    tensor_meta, val=val, mutable=mutable
                )
        elif val is not None:
            # some nodes with symbolic inputs pass a 'val' attribute rather than
            # tensor_meta
            if isinstance(val, TorchFakeTensor):
                return self.get_vtensor_type(
                    val.size(), val.dtype, val=val, mutable=mutable
                )
            elif isinstance(val, list) and all(
                isinstance(x, TorchFakeTensor) for x in val
            ):
                return IrType.parse("!torch.list<vtensor>", context=self._c)

        # Note that None is a valid scalar here, so it is important that this
        # is always checked as the last fallback.
        t = SCALAR_TYPE_TO_TORCH_MLIR_TYPE.get(type(val))
        if t is not None:
            return IrType.parse(t, self._c)

        raise NotImplementedError(
            f"Could not deduce type from value info: "
            f"tensor_meta={tensor_meta}, val={val} {type(val)}, sparsity={sparsity}"
        )

    def tensor_metadata_to_type(
        self,
        tm: TensorMetadata,
        *,
        val: Optional[torch.Tensor] = None,
        mutable: bool = False,
    ) -> IrType:
        tm_shape = tuple(
            item.node if is_symbolic(item) else item for item in list(tm.shape)
        )

        key = (tm_shape, tm.dtype, val, mutable)
        t = self._tensor_metadata_cache.get(key)
        if t is None:
            t = self.get_vtensor_type(tm.shape, tm.dtype, val=val, mutable=mutable)
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

    def create_vtensor_type(self, dtype: torch.dtype, size: torch.Size) -> IrType:
        dtype_asm = str(self.dtype_to_type(dtype))
        return IrType.parse(
            f"!torch.vtensor<{list(size)},{dtype_asm}>", context=self._c
        )

    def tensor_to_vtensor_type(self, tensor: torch.Tensor) -> IrType:
        return self.create_vtensor_type(tensor.dtype, tensor.size())

    def get_node_location(self, node: torch_fx.Node) -> Optional[Location]:
        stack_trace = node.meta.get("stack_trace")
        if stack_trace is None:
            return None
        # Ugh.
        # TODO: Avoid needing to regex match this.
        # https://github.com/pytorch/pytorch/issues/91000
        stack_trace = node.stack_trace
        if stack_trace:
            matches = re.findall(r"""File "([^"]+)", line ([0-9]+),""", stack_trace)
            locations = [
                Location.file(m[0], int(m[1]), col=0, context=self._c) for m in matches
            ]
            if len(locations) > 1:
                return Location.callsite(
                    locations[-1], locations[-2::-1], context=self._c
                )
            elif len(locations) == 1:
                return locations[0]
        return Location.unknown(context=self._c)

    def set_symbolic_guards(
        self, prog: torch.export.ExportedProgram
    ) -> Dict[str, RangeConstraint]:

        # Recent PyTorch versions use `int_oo` to represent integer infinity.
        # Older PyTorch versions like PyTorch stable version may not have
        # `int_oo` defined just yet.
        infs = (sympy.oo, int_oo) if int_oo is not None else (sympy.oo,)

        def _sympy_int_to_int(val: sympy.Expr, adjust_func: Callable):
            # Convert simple sympy Integers into concrete int
            if val in infs:
                return torch.iinfo(torch.int64).max
            if val in tuple(-inf for inf in infs):
                return torch.iinfo(torch.int64).min
            if isinstance(val, sympy.Integer):
                return int(val)
            # TODO: Remove this adjustment when fractional ranges are removed
            return adjust_func(val)

        contains_symbolic_ints = False
        sym_int_types = (
            (sympy.Integer, IntInfinity, NegativeIntInfinity)
            if IntInfinity is not None
            else sympy.Integer
        )
        for val in prog.range_constraints.values():
            if (
                isinstance(val.lower, sym_int_types)
                and isinstance(val.upper, sym_int_types)
                and not val.is_bool
            ):
                contains_symbolic_ints = True
                break
        if contains_symbolic_ints:
            # Build a map from shape symbol name to `RangeConstraint` object
            # capturing `min_val`` and `max_val`` constraints for that
            # symbol. Translate sympy integers to regular integers.
            #
            # Example:
            #       {
            #          's0': RangeConstraint(min_val=5, max_val=10),
            #          's1': RangeConstraint(min_val=0, max_val=100),
            #          's3': RangeConstraint(min_val=0, max_val=9223372036854775806),
            #       }
            self._symbolic_guards = {
                str(k): RangeConstraint(
                    _sympy_int_to_int(v.lower, math.ceil),
                    _sympy_int_to_int(v.upper, math.floor),
                )
                for k, v in prog.range_constraints.items()
            }

    def get_symbolic_guards(self) -> Dict[str, RangeConstraint]:
        return self._symbolic_guards


class GraphNodeImporter:
    """Imports graph nodes into an MLIR function.

    The caller must have already created the function.
    """

    __slots__ = [
        "_b",
        "_c",
        "_cc",
        "_on_node_produced",
        "_v",
        "_symbol_to_value",
        "_multi_result_nodes",
        "_unpack_list_values",
        "fx_importer",
    ]

    def __init__(
        self,
        fx_importer: FxImporter,
        context: Context,
        context_cache: ContextCache,
        block: Block,
    ):
        self.fx_importer = fx_importer
        self._c = context
        self._cc = context_cache
        self._b = block
        # Map of (Node, result_index) to MLIR Value or a callback that lazily
        # constructs and returns a value.
        self._v: Dict[Union[Callable[[], Value], Tuple[torch_fx.Node, int]], Value] = {}
        # Map of Shape Symbol to MLIR Value
        self._symbol_to_value: Dict[str, Value] = {}
        # Map of node name to hook that should be called when it is produced.
        self._on_node_produced: Dict[str, Callable[[Value], None]] = {}
        # Statically multi-result nodes which we have de-tupled are noted here.
        # They will have their getitem calls short-circuited.
        self._multi_result_nodes: Set[torch_fx.Node] = set()
        # If a OP returns a list, then it needs to be unpacked entirely using
        # prim.ListUnpack.  Cache the result of these nodes so that it only
        # unpacks once instead of every time that getitem is used
        self._unpack_list_values: Dict[torch_fx.Node, Tuple[Value]] = {}

    def bind_node_value(
        self,
        node: Node,
        value: Union[Value, Callable[[], Value]],
        result_index: int = 0,
    ):
        """Binds a node to a value (and asserts if already bound).

        This is used by outside callers. Many internal callers poke directly
        into the dict.
        """
        key = (node, result_index)
        assert key not in self._v, f"Node already has a value: {node}"
        self._v[key] = value

        producer_callback = self._on_node_produced.get(node.name)
        if producer_callback is not None:
            producer_callback(value)

    def resolve_node_value(self, node: Node, result_index: int = 0) -> Value:
        """Resolves a node to a value."""
        key = (node, result_index)
        try:
            binding = self._v[key]
        except KeyError:
            raise KeyError(f"FX Node {node} has not been bound to an MLIR value")
        if isinstance(binding, Value):
            return binding

        # It is a lazy callback.
        value = binding()
        self._v[key] = value
        return value

    def bind_symbol_value(
        self,
        shape_symbol: str,
        value: Value,
    ):
        """Binds a shape symbol to a global SSA value (and asserts if already bound)."""
        assert (
            shape_symbol not in self._symbol_to_value
        ), f"Symbol already has a value: {shape_symbol}"
        self._symbol_to_value[shape_symbol] = value

    def resolve_symbol_value(self, shape_symbol: str) -> Value:
        """Resolves a shape symbol to a value."""
        try:
            binding = self._symbol_to_value[shape_symbol]
        except KeyError:
            raise KeyError(
                f"Shape symbol {shape_symbol} has not been bound to an MLIR value"
            )
        if isinstance(binding, Value):
            return binding

    def import_mutable_to_vtensor(
        self, loc: Location, node: Node, mutable_value: Value, producer_node_name: str
    ) -> Value:
        """Imports a node that is represented by a mutable IR value.

        This will generate and associate the following with the node:
          %0 = torch.copy.to_vtensor {mutable_value}

        Then it will also add a trigger such that when `producer_node_name` is
        produced, the following will be generated:
          torch.overwrite.tensor.contents {producer}, {mutable_value}
        """
        with loc, InsertionPoint(self._b):
            immutable_type = self._cc.node_val_to_type(node)
            copy_result = Operation.create(
                "torch.copy.to_vtensor",
                results=[immutable_type],
                operands=[mutable_value],
            ).result
            self.bind_node_value(node, copy_result)

        # Add the producer trigger.
        def on_produced(value: Value):
            with loc, InsertionPoint(self._b):
                Operation.create(
                    "torch.overwrite.tensor.contents",
                    results=[],
                    operands=[value, mutable_value],
                )

        self._on_node_produced[producer_node_name] = on_produced
        return copy_result

    def import_constant(self, loc: Location, node: Node, constant: Any) -> Value:
        with loc, InsertionPoint(self._b):
            value = self._import_literal(constant)
            self.bind_node_value(node, value)
        return value

    def lazy_import_parameter(
        self, loc, node: Node, parameter_value: Any, info: InputInfo
    ):
        def _on_access() -> Value:
            with loc, InsertionPoint(self._b):
                # TODO: Should go to a parameter binding hook.
                return self._import_input(parameter_value, info)

        self.bind_node_value(node, _on_access)

    def lazy_import_buffer(
        self,
        loc,
        node: Node,
        buffer_value: Any,
        info: InputInfo,
    ):
        def _on_access() -> Value:
            with loc, InsertionPoint(self._b):
                # TODO: Should go to a buffer binding hook.
                return self._import_input(buffer_value, info)

        self.bind_node_value(node, _on_access)

        if info.mutable_producer_node_name is not None:
            raise NotImplementedError("NYI: Mutable SSA buffer updates")

        if info.store_producer_node is not None:

            def on_produced(value: Value):
                with loc, InsertionPoint(self._b):
                    self.fx_importer._hooks.store_produced_value(
                        self, buffer_value, value, info
                    )

            self._on_node_produced[info.store_producer_node] = on_produced

    def return_node_values(self, loc, nodes: List[Node], constants: Dict[int, Any]):
        # This function returns both node values and constant values
        with loc, InsertionPoint(self._b):
            operands = [
                (
                    self.resolve_node_value(n)
                    if isinstance(n, Node)
                    else self._import_literal(constants[index])
                )
                for index, n in enumerate(nodes)
            ]
            func_dialect.ReturnOp(operands, loc=loc)

    def import_nodes(
        self,
        nodes: Iterable[Node],
        *,
        skip_placeholders_outputs: bool = False,
        import_symbolic_shape_expressions: bool = False,
    ):
        with InsertionPoint(self._b):
            loc = Location.unknown()

            # Import dynamic shape symbols and guards (if any)
            if import_symbolic_shape_expressions:
                symbolic_guards = self._cc.get_symbolic_guards()
                self._import_shape_symbols_with_guards(loc, symbolic_guards)

            num_placeholders = 0
            for node in nodes:
                op = node.op
                # Attempt to extract locations. Not everything has them,
                # so we do our best.
                new_loc = self._cc.get_node_location(node)
                if new_loc is not None:
                    loc = new_loc
                if op == "placeholder" and not skip_placeholders_outputs:
                    # Associate the placeholder node with corresponding block
                    # argument.
                    self.bind_node_value(node, self._b.arguments[num_placeholders])
                    num_placeholders += 1
                elif op == "call_function":
                    target = node.target
                    if target == operator.getitem:
                        self._import_getitem(loc, node)
                    elif target in SYMBOLIC_TORCH_OPS or (
                        is_symbolic(node.meta.get("val"))
                        and is_builtin_function_or_method(target)
                    ):
                        self._import_symbolic_torch_op(loc, node, target)
                    elif isinstance(target, TorchOpOverload):
                        # Dispatch to an ATen op.
                        self._import_torch_op_overload(loc, node)
                    elif isinstance(target, HigherOrderOperator):
                        self._import_hop(loc, node, target)
                    else:
                        raise NotImplementedError(
                            f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                        )
                elif op == "output" and not skip_placeholders_outputs:
                    # args[0] is a singleton tuple that we flatten into multiple
                    # results.
                    output_arg = node.args[0]
                    # Handle both single Node and tuple/list of Nodes
                    result_nodes = (
                        output_arg
                        if isinstance(output_arg, (list, tuple))
                        else [output_arg]
                    )
                    operands = [self._import_argument(loc, arg) for arg in result_nodes]
                    func_dialect.ReturnOp(operands, loc=loc)

                if import_symbolic_shape_expressions:
                    self._create_bind_symbolic_shape_ops(loc, node)

    def _promote_symbolic_scalar_int_float(self, loc, graph, param):
        temp_target = torch.ops.aten.Float.Scalar
        temp_node = Node(
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
                (arg.meta["val"].node.pytype if isinstance(arg, Node) else type(arg))
                for arg in node.args
            ]
            is_int = [item is int for item in arg_types]
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
                        if isinstance(arg0, Node):
                            prom_arg = self._promote_symbolic_scalar_int_float(
                                loc, node.graph, arg0
                            )
                            new_args = (prom_arg, arg1)
                        else:
                            arg0 = float(arg0)
                            new_args = (arg0, arg1)
                    else:
                        if isinstance(arg1, Node):
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
            if _IS_TORCH_2_1_OR_EARLIER:
                concrete_target = SYMBOLIC_OP_TO_TORCH_OP.get((target, len(node.args)))
            else:
                concrete_target = SYMBOLIC_OP_TO_TORCH_OP.get(target)

        assert (
            concrete_target is not None
        ), f"Unable to parse symbolic operation: {target} with args {node.args}"
        self._import_torch_op_overload(loc, node, concrete_target)

    def _import_hop(self, loc: Location, node: torch_fx.Node, hop: HigherOrderOperator):
        # Imports a higher-order operator.
        # See: https://dev-discuss.pytorch.org/t/higher-order-operators-2023-10/1565
        assert hop.namespace == "higher_order"
        hop_name = hop.name()
        handler_name = f"_import_hop_{hop_name}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise NotImplementedError(
                f"Higher-order operation '{hop_name}' not "
                f"implemented in the FxImporter "
                f"(tried '{handler_name}')"
            )
        handler(loc, node, hop)

    def _import_hop_while_loop(
        self, loc: Location, node: torch_fx.Node, hop: HigherOrderOperator
    ):
        """Imports the torch._higher_order_ops.while_loop HOP.

        Args format: (cond_fn, body_fn, carries)
        The cond_fn and body_fn are get_attr nodes pointing to submodule graphs
        that have already been imported by import_program().

        Emits torch.prim.Loop with proper control flow structure.
        """
        # while_loop HOP args: (cond_fn, body_fn, car`ries...)
        # Unpack the first two args and the rest as carries
        cond_fn_node, body_fn_node, *carries = node.args

        # Extract function names from get_attr nodes
        assert (
            cond_fn_node.op == "get_attr"
        ), f"Expected get_attr for cond_fn, got {cond_fn_node.op}"
        assert (
            body_fn_node.op == "get_attr"
        ), f"Expected get_attr for body_fn, got {body_fn_node.op}"

        root_module = node.graph.owning_module
        cond_fn_module = getattr(root_module, cond_fn_node.target, None)
        body_fn_module = getattr(root_module, body_fn_node.target, None)

        # Generate function names with module IDs for uniqueness
        cond_fn_name = self.fx_importer._graph_module_to_func_name[id(cond_fn_module)]
        body_fn_name = self.fx_importer._graph_module_to_func_name[id(body_fn_module)]

        # Import the carries (loop state variables)
        carry_values = []
        for carry in carries:
            if isinstance(carry, tuple):
                # Handle tuple carries by importing each element
                carry_values.extend(self._import_tuple_argument(loc, carry, None))
            else:
                carry_values.append(self._import_argument(loc, carry))

        # Determine result types from node metadata
        node_val = node.meta.get("val")
        if isinstance(node_val, (list, tuple)) and len(node_val) > 1:
            result_types = [self._cc.value_info_to_type(v) for v in node_val]
            self._multi_result_nodes.add(node)
        else:
            result_types = [self._cc.node_val_to_type(node)]

        # Call the condition function with initial carries to get initial condition
        cond_result_type = self._cc.get_vtensor_type(torch.Size([]), torch.bool)

        initial_cond_call = Operation.create(
            "func.call",
            attributes={"callee": FlatSymbolRefAttr.get(cond_fn_name)},
            results=[cond_result_type],
            operands=carry_values,
            loc=loc,
        )

        # Convert vtensor<bool> to torch.bool
        bool_conv = Operation.create(
            name="torch.aten.Bool.Tensor",
            results=[self._cc.torch_bool_type],
            operands=[initial_cond_call.results[0]],
            loc=loc,
        )

        # Create max iterations constant (INT64_MAX)
        with loc:
            max_iter = _make_constant_op(
                "torch.constant.int",
                self._cc.integer_attr(torch.iinfo(torch.int64).max, 64),
                self._cc.torch_int_type,
            )

        # Create torch.prim.Loop operation with region
        loop_op = Operation.create(
            name="torch.prim.Loop",
            results=result_types,
            operands=[max_iter.results[0], bool_conv.results[0]] + carry_values,
            regions=1,
            loc=loc,
        )

        # Create loop body region with block arguments
        # Block args: iteration counter (!torch.int) + all carry values
        loop_region = loop_op.regions[0]
        block_arg_types = [self._cc.torch_int_type] + result_types
        with loc:
            loop_block = Block.create_at_start(loop_region, block_arg_types)

        # Inside the loop body, call body function and condition function
        with InsertionPoint(loop_block):
            # Call body function with current carry values (skip iteration counter)
            body_results_op = Operation.create(
                name="func.call",
                attributes={"callee": FlatSymbolRefAttr.get(body_fn_name)},
                results=result_types,
                operands=list(loop_block.arguments[1:]),  # Skip iteration counter
                loc=loc,
            )
            body_results = list(body_results_op.results)

            # Call condition function with updated carries
            cond_result_loop = Operation.create(
                name="func.call",
                attributes={"callee": FlatSymbolRefAttr.get(cond_fn_name)},
                results=[IrType.parse("!torch.vtensor<[],i1>", context=self._c)],
                operands=body_results,
                loc=loc,
            ).result

            # Convert to bool
            cond_bool = Operation.create(
                name="torch.aten.Bool.Tensor",
                results=[self._cc.torch_bool_type],
                operands=[cond_result_loop],
                loc=loc,
            ).result

            # Emit loop condition with updated carries
            Operation.create(
                name="torch.prim.Loop.condition",
                results=[],
                operands=[cond_bool] + body_results,
                loc=loc,
            )

        # Bind the loop results to the node
        if len(result_types) > 1:
            self._multi_result_nodes.add(node)
            for i, value in enumerate(loop_op.results):
                self.bind_node_value(node, value, i)
        else:
            self.bind_node_value(node, loop_op.results[0])

    def _import_hop_auto_functionalized(
        self, loc: Location, node: torch_fx.Node, hop: HigherOrderOperator
    ):
        # Imports the torch._higher_order_ops.auto_functionalize.auto_functionalized HOP.
        # This op wraps a target OpOverload with args/kwargs dispatched to it.
        # Even thought the OpOverload will return None, this returns the
        # arguments mutated. Note that the general op overload importing can't
        # be used here as they use a special encoding for everything.
        # See: torch/_higher_order_ops/auto_functionalize.py
        (op_overload,) = node.args
        schema = op_overload._schema
        assert isinstance(schema, FunctionSchema)
        mlir_op_name = _get_mlir_op_name_for_schema(schema)

        # Functionalization transforms the results to (*actual, *aliased).
        # If the schema is actually zero return, then the first "val"
        # type will be None and we need to bind that as a result of the node.
        # However, that doesn't make it into the IR. This special casing is
        # annoying.
        node_result_types = [
            (None if v is None else self._cc.tensor_metadata_to_type(v))
            for v in node.meta["val"]
        ]

        if len(schema.returns) == 0:
            assert node_result_types[0] is None
            ir_result_types = node_result_types[1:]
            bind_none = 1
        else:
            ir_result_types = node_result_types
            bind_none = 0

        # The auto_functionalized ops maps all arguments by name (as opposed
        # to mixed for generic OpOverload). Linearize them.
        operands = []
        for parameter in schema.arguments:
            operand = self._import_argument(
                loc, node.kwargs[parameter.name], parameter.type
            )
            operands.append(operand)

        operation = _emit_operation(
            mlir_op_name,
            result_types=ir_result_types,
            operands=operands,
            loc=loc,
        )

        # Special case: if declared_result_types was empty, then we bind a
        # None for future node access.
        self._multi_result_nodes.add(node)
        if bind_none:
            self.bind_node_value(node, None, 0)
        # Record value mappings for remainder.
        for i, value in enumerate(operation.results):
            self.bind_node_value(node, value, i + bind_none)

    def _import_hop_flex_attention(
        self, loc: Location, node: torch_fx.Node, hop: HigherOrderOperator
    ):
        """Imports the torch._higher_order_ops.flex_attention HOP.

        Args format: (query, key, value, score_mod, block_mask, scale, enable_gqa, kernel_options, ...)
        - query, key, value: Attention input tensors
        - score_mod: Optional submodule/callable for score modification (imported as function)
        - block_mask: Optional BlockMask tuple containing mask_mod function and runtime tensors
        - scale: Optional float for attention score scaling
        - enable_gqa: Boolean for grouped query attention support (TODO: NYI)
        - kernel_options: Dict of performance tuning options (TODO: NYI)

        This creates a call to aten.flex_attention with function symbol references for
        score_mod and mask_mod.
        """
        # flex_attention HOP args from PyTorch:
        # (query, key, value, score_mod, block_mask, scale, enable_gqa, kernel_options, return_lse_tuple, ...)
        if len(node.args) < 6:
            raise ValueError(
                f"flex_attention expects at least 6 arguments, got {len(node.args)}"
            )

        query_arg, key_arg, value_arg, score_mod_arg, block_mask_arg, scale_arg = (
            node.args[:6]
        )

        # TODO: Add support for enable_gqa (grouped query attention)
        # This is a boolean flag that enables GQA optimization
        enable_gqa = node.args[6] if len(node.args) > 6 else False

        # TODO: Add support for kernel_options (performance tuning parameters)
        # This is a dict containing options like block sizes, num_warps, etc.
        kernel_options = node.args[7] if len(node.args) > 7 else {}

        # Import Q, K, V tensors
        query = self._import_argument(loc, query_arg, None)
        key = self._import_argument(loc, key_arg, None)
        value = self._import_argument(loc, value_arg, None)

        score_mod_ref = None
        if score_mod_arg is not None and isinstance(score_mod_arg, torch_fx.Node):
            assert (
                score_mod_arg.op == "get_attr"
            ), f"Expected get_attr for score_mod, got {score_mod_arg.op}"
            root_module = node.graph.owning_module
            score_mod_module = getattr(root_module, score_mod_arg.target, None)
            if score_mod_module is not None:
                score_mod_func_name = self.fx_importer._graph_module_to_func_name[
                    id(score_mod_module)
                ]
                score_mod_ref = FlatSymbolRefAttr.get(score_mod_func_name)

        # Handle block_mask: extract only mask_mod function reference
        # Note: BlockMask contains runtime tensors (kv_num_blocks, kv_indices, etc.)
        # that are materialized by evaluating mask_mod(b, h, q_idx, kv_idx).
        mask_mod_ref = None
        if block_mask_arg is not None and isinstance(block_mask_arg, tuple):
            root_module = node.graph.owning_module
            # The mask_mod function is the last element in the BlockMask tuple
            mask_mod_arg = block_mask_arg[-1]
            if mask_mod_arg is not None and isinstance(mask_mod_arg, torch_fx.Node):
                assert (
                    mask_mod_arg.op == "get_attr"
                ), f"Expected get_attr for mask_mod, got {mask_mod_arg.op}"
                mask_mod_module = getattr(root_module, mask_mod_arg.target, None)
                if mask_mod_module is not None:
                    mask_mod_func_name = self.fx_importer._graph_module_to_func_name[
                        id(mask_mod_module)
                    ]
                    mask_mod_ref = FlatSymbolRefAttr.get(mask_mod_func_name)

        # Import scale (float or None)
        if scale_arg is None:
            scale = Operation.create(
                "torch.constant.none",
                results=[self._cc.torch_none_type],
                loc=loc,
            ).result
        elif isinstance(scale_arg, (int, float)):
            with loc:
                scale = _make_constant_op(
                    "torch.constant.float",
                    FloatAttr.get_f64(float(scale_arg)),
                    self._cc.torch_float_type,
                ).result
        else:
            scale = self._import_argument(loc, scale_arg, None)

        # Determine result types from node metadata
        node_val = node.meta.get("val")
        if isinstance(node_val, (list, tuple)) and len(node_val) >= 2:
            # flex_attention returns (output, logsumexp)
            result_types = [self._cc.value_info_to_type(v) for v in node_val]
            self._multi_result_nodes.add(node)
        else:
            # Single output
            result_types = [self._cc.node_val_to_type(node)]

        with loc:
            enable_gqa_value = _make_constant_op(
                "torch.constant.bool",
                self._cc.integer_attr(1 if enable_gqa else 0, 1),
                self._cc.torch_bool_type,
            ).result

        with loc:
            return_lse = _make_constant_op(
                "torch.constant.bool",
                self._cc.integer_attr(
                    (
                        1
                        if (
                            getattr(node_val, "return_lse", False)
                            or (
                                isinstance(node_val, (list, tuple))
                                and len(node_val) >= 2
                            )
                        )
                        else 0
                    ),
                    1,
                ),
                self._cc.torch_bool_type,
            ).result

        # Build operands for aten.flex_attention.
        # Op expects exactly 5 operands: query, key, value, scale, return_lse.
        # Note: score_mod_fn and mask_mod_fn go as ATTRIBUTES, not operands.
        # Note: block_mask tensors are handled by mask_mod_fn, not passed as operands.

        flat_operands = [
            query,
            key,
            value,
            scale,
            enable_gqa_value,
            return_lse,
        ]

        # Build attributes with function references
        # Only include attributes if they're not None (OptionalAttr in TableGen)
        attributes = {}
        if score_mod_ref is not None:
            attributes["score_mod_fn"] = score_mod_ref
        if mask_mod_ref is not None:
            attributes["mask_mod_fn"] = mask_mod_ref

        operation = Operation.create(
            "torch.aten.flex_attention",
            results=result_types,
            operands=flat_operands,
            attributes=attributes if attributes else None,
            loc=loc,
        )

        # Bind results
        if len(result_types) > 1:
            self._multi_result_nodes.add(node)
            for i, value in enumerate(operation.results):
                self.bind_node_value(node, value, i)
        else:
            self.bind_node_value(node, operation.results[0])

    def _import_torch_op_overload(
        self,
        loc: Location,
        node: torch_fx.Node,
        concrete_target: Optional[TorchOpOverload] = None,
    ):
        if concrete_target is None:
            node = node_canonicalize(node)
            if not node:
                return
            target = node.target
        else:
            target = concrete_target

        schema = target._schema
        assert isinstance(schema, FunctionSchema)
        mlir_op_name = _get_mlir_op_name_for_schema(schema)

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

        # Convert result types.
        result_types = self._unpack_node_result_types(node, schema)
        if len(result_types) > 1:
            self._multi_result_nodes.add(node)

        # Unroll operands from formal parameters, args and kwargs.
        operands = []
        for i, parameter in enumerate(schema.arguments):
            if i < len(node.args):
                operands.append(
                    self._import_argument(loc, node.args[i], parameter.type)
                )
            elif parameter.name in node.kwargs:
                operands.append(
                    self._import_argument(
                        loc, node.kwargs[parameter.name], parameter.type
                    )
                )
            else:
                operands.append(
                    self._import_default_value(
                        loc, parameter.default_value, parameter.type
                    )
                )

        operation = _emit_operation(
            mlir_op_name, result_types=result_types, operands=operands, loc=loc
        )

        # Record value mapping.
        for i, value in enumerate(operation.results):
            self.bind_node_value(node, value, i)

    def _import_shape_symbols_with_guards(
        self, loc: Location, symbolic_guards: Dict[str, RangeConstraint]
    ):
        for symbol, constraints in symbolic_guards.items():
            # Create torch.sym_int ops
            operation = Operation.create(
                name="torch.symbolic_int",
                attributes={
                    "symbol_name": StringAttr.get(symbol),
                    "min_val": self._cc.integer_attr(constraints.min_val, 64),
                    "max_val": self._cc.integer_attr(constraints.max_val, 64),
                },
                results=[self._cc.torch_int_type],
                loc=loc,
            )
            self.bind_symbol_value(symbol, operation.result)

    def _create_bind_symbolic_shape_ops(self, loc: Location, node: torch_fx.Node):
        node_val = node.meta.get("val")
        if (node_val is not None) and isinstance(node_val, TorchFakeTensor):
            # Only create bind ops if the shapes contain symbolic sizes.
            # Query the bool attribute `_has_symbolic_sizes_strides` on node.meta["val"].
            if node_val._has_symbolic_sizes_strides:
                # Read node metadata to obtain shape symbols and expressions
                symbols_set = set()
                shape_exprs = []
                for s in node_val.size():
                    if isinstance(s, torch.SymInt):
                        symbols_set.update(s.node.expr.free_symbols)
                        shape_exprs.append(s.node.expr)
                    else:
                        assert isinstance(s, int)
                        shape_exprs.append(s)

                # Map from sympy shape symbols to local symbols in the affine map
                symbols_set = sorted(symbols_set, key=lambda x: x.name)
                symbols_map = {
                    str(symbol): AffineSymbolExpr.get(i)
                    for i, symbol in enumerate(symbols_set)
                }

                # Convert symbolic shape expressions into affine expressions
                affine_exprs = [
                    sympy_expr_to_semi_affine_expr(expr, symbols_map)
                    for expr in shape_exprs
                ]

                affine_map = AffineMap.get(0, len(symbols_set), affine_exprs)

                # Build operand list
                operand_list = []
                operand_list.append(self.resolve_node_value(node))
                for symbol in symbols_map.keys():
                    operand_list.append(self.resolve_symbol_value(symbol))

                # Create torch.bind_symbolic_shape ops
                Operation.create(
                    name="torch.bind_symbolic_shape",
                    attributes={"shape_expressions": AffineMapAttr.get(affine_map)},
                    operands=operand_list,
                    loc=loc,
                )

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
                    self.bind_node_value(arg, self._import_literal(obj))

            argument_value = self.resolve_node_value(arg)
        elif isinstance(arg, torch_fx.immutable_collections.immutable_list):
            argument_value = self._import_list_argument(loc, arg, expected_jit_type)
        elif isinstance(arg, tuple):
            # Handle tuples of tensors (common in while_loop carries)
            argument_value = self._import_tuple_argument(loc, arg, expected_jit_type)
        elif isinstance(expected_jit_type, torch.TensorType) and not isinstance(
            arg, torch.Tensor
        ):
            # promote scalars to tensor types as appropriate
            argument_value = self._import_scalar_as_tensor(loc, arg)
        elif LITERAL_CONVERTER_MAP.lookup(type(arg)) is not None:
            with loc:
                argument_value = self._import_literal(arg)
        else:
            raise TypeError(f"Unsupported argument type {arg.__class__}")
        with loc:
            return self._convert_type(argument_value, expected_jit_type)

    def _convert_type(
        self,
        val: Value,
        expected_type,
        dtype: Optional[torch.dtype] = None,
        size: Optional[torch.Size] = None,
    ):
        """
        When the type of 'value' and the type in the schema do not match,
        attempt to perform automatic type conversion.

        example: test/python/fx_importer/basic_test.py::test_full
        """
        if not expected_type:
            return val
        op_name = None
        result_type = None
        # TODO: If additional types require conversion in the future,
        #  consider implementing a table-driven approach.
        operands = [val]
        if val.type == self._cc.torch_bool_type:
            if isinstance(expected_type, torch.FloatType):
                op_name = "torch.aten.Float.bool"
                result_type = self._cc.torch_float_type
            elif isinstance(expected_type, (torch.IntType, torch.NumberType)):
                op_name = "torch.aten.Int.bool"
                result_type = self._cc.torch_int_type
        elif expected_type is torch.Tensor:
            op_name = "torch.prims.convert_element_type"
            result_type = self._cc.create_vtensor_type(dtype, size)
            operands.append(
                LITERAL_CONVERTER_MAP.lookup(torch.dtype)(dtype, self, self._cc)
            )
        if op_name is None:
            return val
        return Operation.create(
            name=op_name, results=[result_type], operands=operands
        ).result

    def _import_literal(self, py_value: Any, info: Optional[InputInfo] = None) -> Value:
        orig_value = None
        if isinstance(py_value, torch.Tensor) and py_value.dtype == torch.bool:
            orig_value = py_value
            py_value = py_value.to(torch.uint8)
        # Apply the conversion callback.
        user_value = self.fx_importer._hooks.resolve_literal(self, py_value, info)
        if user_value is not None:
            assert isinstance(user_value, Value)
            if orig_value is not None:
                user_value = self._convert_type(
                    user_value, torch.Tensor, orig_value.dtype, orig_value.size()
                )
            return user_value

        # Default conversion path.
        converter = LITERAL_CONVERTER_MAP.lookup(type(py_value))
        if converter is None:
            raise TypeError(
                f"Unsupported argument -> literal conversion for {py_value.__class__}"
            )
        result = converter(py_value, self, self._cc)
        if orig_value is not None:
            result = self._convert_type(
                result, torch.Tensor, orig_value.dtype, orig_value.size()
            )
        return result

    def _import_input(self, py_value: Any, info: InputInfo) -> Value:
        # Try the hook.
        user_value = self.fx_importer._hooks.resolve_input(self, py_value, info)
        if user_value is not None:
            assert isinstance(user_value, Value)
            return user_value

        # Fall-back to treating as a literal if not mutating.
        if info.mutable_producer_node_name is not None:
            raise ValueError(
                f"Cannot import {info.input_spec} as a literal because it is mutable"
            )
        return self._import_literal(py_value, info)

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

    def _import_tuple_argument(
        self, loc: Location, arg: tuple, expected_jit_type
    ) -> list[Value]:
        """Import a tuple argument by importing each element separately."""
        # For tuples in while_loop carries, treat each element as a separate argument
        return [self._import_argument(loc, elem, expected_jit_type) for elem in arg]

    def _import_list_argument(
        self, loc: Location, arg: Sequence[NodeArgument], expected_jit_type
    ) -> Value:
        assert (
            isinstance(expected_jit_type, torch.ListType)
            or (
                isinstance(expected_jit_type, torch.OptionalType)
                and isinstance(expected_jit_type.getElementType(), torch.ListType)
            )
            or (expected_jit_type is None)
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
            if isinstance(operand, Node):
                if operand in self._multi_result_nodes:
                    raise RuntimeError(f"Attempt to de-reference a multi-result node")
                val = self.resolve_node_value(operand)
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

    def _import_getitem(self, loc: Location, node: torch.fx.Node):
        ref_node, index = node.args
        if ref_node in self._multi_result_nodes:
            # Special case handling of getitem for when it is resolving
            # against a function call that we know has returned multiple
            # results. We short-circuit this case because we have modeled
            # function calls to natively return multiple results vs tupling.
            try:
                self.bind_node_value(
                    node,
                    self.resolve_node_value(ref_node, index),
                )
            except IndexError:
                raise RuntimeError(
                    f"getitem de-aliasing failed. This likely "
                    f"indicates a programmer error that usually "
                    f"would have happened at runtime. Please "
                    f"notify developers if this case happens "
                    f"(at {loc})."
                )
        else:
            # handle nodes that return a torch.list<...> at the MLIR level
            # NOTE: the length of the list must be knowable at compile time.
            if ref_node not in self._unpack_list_values:
                node_result = self.resolve_node_value(ref_node, 0)

                if str(node_result.type) in TORCH_LIST_TYPES:
                    result_types = [
                        self._cc.value_info_to_type(v) for v in ref_node.meta["val"]
                    ]
                    operation = Operation.create(
                        "torch.prim.ListUnpack",
                        results=result_types,
                        operands=[node_result],
                        loc=loc,
                    )
                    self._unpack_list_values[ref_node] = tuple(operation.results)

            try:
                self.bind_node_value(node, self._unpack_list_values[ref_node][index])
            except IndexError:
                raise RuntimeError(
                    f"getitem failed. "
                    f"getitem only supports lists of known length. (at {loc})"
                )

    def _unpack_node_result_types(
        self, node: torch.fx.Node, schema: FunctionSchema
    ) -> List[IrType]:
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
                result_types.append(self._cc.value_info_to_type(v))
        return result_types


def _make_constant_op(
    op_name: str, value_attr: Attribute, result_type: Optional[IrType] = None
) -> Operation:
    return Operation.create(
        op_name,
        results=[result_type if result_type else value_attr.type],
        attributes={"value": value_attr},
    )


def _create_mlir_tensor_type(dtype: torch.dtype, size: torch.Size) -> IrType:
    try:
        element_type = TORCH_DTYPE_TO_MLIR_TYPE[dtype]()
        tensor_type = RankedTensorType.get(size, element_type)
        return tensor_type
    except KeyError:
        raise TypeError(f"Could not map Torch dtype {dtype} to an MLIR type")


def create_mlir_tensor_type(tensor: torch.Tensor) -> IrType:
    return _create_mlir_tensor_type(tensor.dtype, tensor.size())


def _make_vtensor_literal_op(
    tensor: torch.Tensor, vtensor_type: IrType, py_attr_tracker: "RefTracker"
) -> Operation:
    mapping = py_attr_tracker.track(tensor)
    if mapping.is_empty:
        # check support for bfloat16
        assert not (
            tensor.dtype == torch.bfloat16 and ml_dtypes is None
        ), f"torch.bfloat16 requires the ml_dtypes package, please run:\n\npip install ml_dtypes\n"
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
        # One element constants are more optimizable as splat DenseElementsAttr. DenseResourceElementsAttr does not
        # support splats, so don't use it for that case. In addition, at the time of writing, it has bugs with handling
        # 0d tensors.
        if np_tensor.size == 1:
            try:
                dtype = tensor.dtype
                element_type = TORCH_DTYPE_TO_MLIR_TYPE[dtype]()
            except KeyError:
                raise TypeError(f"Could not map Torch dtype {dtype} to an MLIR type")
            elements_attr = DenseElementsAttr.get(
                type=element_type, array=np_tensor, shape=np_tensor.shape
            )
        else:
            bytes_view = np_tensor.view(npy_dtype)
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
# Utilities
###############################################################################


def _get_mlir_op_name_for_schema(schema: FunctionSchema) -> str:
    # Returns a fully-qualified MLIR operation name (i.e. 'torch.foobar')
    # for a function schema.
    namespace, sep, unqualified_name = schema.name.partition("::")
    assert sep, f"Malformed Torch op name {schema.name}"
    mlir_op_name = f"torch.{namespace}.{unqualified_name}"
    if schema.overload_name != "":
        mlir_op_name += f".{schema.overload_name}"
    return mlir_op_name


def _emit_operation(
    mlir_op_name: str, result_types: List[IrType], operands: List[Value], loc: Location
) -> Operation:
    # Support unregistered torch ops using torch.operator.
    # torch.operator is used to represent ops from registry
    # which haven't been generated by torch_ods_gen.py.
    context = loc.context
    if not context.is_registered_operation(mlir_op_name):
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
    return operation


###############################################################################
# Reference mapping
###############################################################################


# Opaque value to indicate something is empty. Used in cases where 'None'
# may have a different meaning.
class EmptyType: ...


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
        # Finalizer is removed due to a memory leak
        # See: https://github.com/iree-org/iree-turbine/issues/281
        # if referrent is not Empty:
        #    weakref.finalize(referrent, self._ref_finalizer, ref_id)
        self._refs[ref_id] = info
        return info

    # def _ref_finalizer(self, ref_id: int):
    #    del self._refs[ref_id]


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

TORCH_LIST_TYPES = set(PY_TYPE_TO_TORCH_LIST_TYPE.values()) | set(
    PY_TYPE_TO_TORCH_OPTIONAL_LIST_TYPE.values()
)

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


NODE_CANONICALIZE: Dict[TorchOpOverload, Callable] = {}


def register_canonicalize(op: TorchOpOverload):
    def wrapper(func):
        NODE_CANONICALIZE[op] = func
        return func

    return wrapper


@register_canonicalize(torch.ops.aten.lift_fresh_copy.default)
def lift_fresh_copy_default(node: torch_fx.Node):
    # replace lift_fresh_copy with clone op
    node.target = torch.ops.aten.clone.default
    node.args = (node.args[0],)
    node.kwargs = {"memory_format": None}
    return node


@register_canonicalize(torch.ops.aten.lift_fresh_copy.out)
def lift_fresh_copy_out(node: torch_fx.Node):
    # TODO: It seems not possible to hit this case from user code.
    # Retaining in case if it is triggered internally somehow, but
    # it can most likely be removed once assuming full
    # functionalization in all cases.
    node.target = target = torch.ops.aten.clone.out
    node.args = (node.args[0],)
    node.kwargs = {"memory_format": None, "out": node.args[1]}
    return node


@register_canonicalize(torch.ops.aten.empty.memory_format)
def empty_memory_format(node: torch_fx.Node):
    # TODO: generalize empty.memory_format in the future
    # Currently, the aten.baddbmm.default op for Unet includes multiplying an
    # empty.memory_format input with a constant, which creates NaN values
    # because empty.memory_format contains uninitialized data. Converting
    # aten.baddbmm.default -> aten.zeros.default fixes the correctness issue
    if len(node.users) == 1:
        for key_node in node.users:
            if key_node.target == torch.ops.aten.baddbmm.default:
                node.target = torch.ops.aten.zeros.default
    return node


@register_canonicalize(torch.ops.aten._local_scalar_dense.default)
def aten__local_scalar_dense_default(node: torch_fx.Node):
    input_type = node.args[0].meta["tensor_meta"].dtype
    if input_type.is_floating_point:
        node.target = torch.ops.aten.Float.Tensor
    else:
        node.target = torch.ops.aten.Int.Tensor
    node.args = (node.args[0],)
    return node


@register_canonicalize(torch.ops.aten._assert_async.msg)
def aten__assert_async_msg(node: torch_fx.Node):
    # TODO: A more suitable op to replace it?
    return None


@register_canonicalize(torch.ops.aten._unsafe_index_put.default)
def aten__unsafe_index_put_default(node: torch_fx.Node):
    node.target = torch.ops.aten._unsafe_index_put.hacked_twin
    return node


@register_canonicalize(torch.ops.aten._embedding_bag_forward_only.default)
def aten__embedding_bag_forward_only_default(node: torch_fx.Node):
    node.target = torch.ops.aten.embedding_bag.padding_idx
    embedding_bag_args = [
        ("scale_grad_by_freq", False),
        ("mode", 0),
        ("sparse", False),
        ("per_sample_weights", None),
        ("include_last_offset", False),
        ("padding_idx", None),
    ]
    node_kwargs = dict(node.kwargs)
    for k, v in embedding_bag_args[len(node.args) - 3 :]:
        if k not in node_kwargs:
            node_kwargs[k] = v
    node.kwargs = node_kwargs
    return node


def node_canonicalize(node: torch_fx.Node):
    if node.target in NODE_CANONICALIZE:
        return NODE_CANONICALIZE[node.target](node)
    return node
