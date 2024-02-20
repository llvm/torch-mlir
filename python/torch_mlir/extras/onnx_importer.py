# Based on code Copyright (c) Advanced Micro Devices, Inc.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Imports ONNX graphs to `torch` dialect ops.

See documentation:
    https://github.com/llvm/torch-mlir/blob/main/docs/importers/onnx_importer.md

This file is distributed/forked verbatim into various downstream projects, and
it must abide by several rules above and beyond the rest of the codebase:
    - It must be standalone, only depending on:
        - `onnx`
        - `..ir` relative imports to the main IR directory
        - `..dialects.func` relative import to the `func` dialect (TODO:
           we are looking to eliminate this dep).
        - Python standard library
    - It does not directly use the ODS generated `torch` dialect Python
      wrappers. This allows it to be used in contexts that only build a C++
      compiler with minimal IR Python bindings.
    - It is intended as an enabler for full onnx compilation, only handling
      the import from ONNX -> the `torch` dialect. Testing, full pipelines,
      and utilities belong elsewhere.
"""

try:
    import onnx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The onnx package (`pip install onnx`) is required to use the onnx importer"
    ) from e

from typing import Optional

from dataclasses import dataclass

import numpy as np
import re

from ..ir import (
    ArrayAttr,
    Attribute,
    Block,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    DictAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3FNType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    MLIRError,
    RankedTensorType,
    Location,
    Module,
    Operation,
    StringAttr,
    Type as IrType,
    Value,
)

from ..dialects import (
    func as func_dialect,
)

@dataclass
class Config:
    """Various configuration settings for the importer."""

    # Ancient ONNX exporters would often add a model input for anything that
    # might be mutable, providing an initializer for it as well. More modern
    # tools tools realized this is a really bad idea for a lot of reasons.
    # We choose to assume more recent norms, even if encountering older
    # models. Setting this to False probably won't do what you want but
    # should produce interesting errors to waste your time deciphering.
    # We mainly use it as a way to document in the code that we are
    # making an assumption.
    elide_initialized_inputs: bool = True


class ModelInfo:
    """Top-level accounting and accessors for an ONNX model."""

    def __init__(self, model_proto: onnx.ModelProto, *, config: Config = Config()):
        self.config = config
        self.model_proto = model_proto
        assert model_proto.graph, "Model must contain a main Graph"
        self.main_graph = GraphInfo(self, model_proto.graph)

    def create_module(self, context: Optional[Context] = None) -> Operation:
        if not context:
            context = Context()
        module_op = Module.create(Location.unknown(context))
        # TODO: Populate module level metadata from the ModelProto
        return module_op


class GraphInfo:
    """Information about a Graph within a model."""

    def __init__(self, model_info: ModelInfo, graph_proto: onnx.GraphProto):
        self.model_info = model_info
        self.graph_proto = graph_proto
        self.initializer_map: dict[str, onnx.TensorProto] = {
            n.name: n for n in graph_proto.initializer
        }
        self.value_info_map: dict[str, onnx.ValueInfoProto] = {
            n.name: n for n in graph_proto.value_info
        }
        self.declared_input_map: dict[str, onnx.ValueInfoProto] = {
            n.name: n for n in graph_proto.input
        }
        self.output_map: dict[str, onnx.ValueInfoProto] = {
            n.name: n for n in graph_proto.output
        }

        # Generate the effective input map, which for old models can be a
        # subset of the input map.
        if model_info.config.elide_initialized_inputs:
            self.input_map = {
                k: v
                for k, v in self.declared_input_map.items()
                if k not in self.initializer_map
            }
        else:
            self.input_map = self.declared_input_map
            illegal_input_keys = self.input_map.keys() - (
                self.input_map.keys() - self.initializer_map.keys()
            )
            assert self.input_map.keys().isdisjoint(self.initializer_map.keys()), (
                f"When not in elide_initialized_inputs=True, we expect inputs to not "
                f"have an initial value (got {illegal_input_keys})."
            )

    def find_type_proto_for_name(self, name: str) -> onnx.TypeProto:
        # Node outputs don't typically have type information, but shape inference
        # will associate them in the value_info. If not there, it may be a
        # graph output, which must have type information.
        value_info = self.value_info_map.get(name) or self.output_map.get(name)
        if value_info is not None:
            return value_info.type
        raise OnnxImportError(
            f"No type information associated with '{name}'. Run shape inference?"
        )


class OnnxImportError(Exception):
    ...


class NodeImporter:
    """Imports graph nodes into MLIR.

    Typically, the top level graph will be imported into a func whereas dependent
    graphs may just be imported with references to pre-existing values.

    Note that ONNX requires that graphs be sorted topologically and free of cycles,
    so we don't take any special steps to order them for dominance.
    """

    __slots__ = [
        "_c",
        "_cc",
        "_gi",
        "_p",
        "_b",
        "_nv_map",
    ]

    def __init__(
        self,
        graph_info: GraphInfo,
        *,
        parent_op: Operation,
        block: Block,
        context_cache: "ContextCache",
    ):
        self._c = parent_op.context
        self._cc = context_cache
        self._gi = graph_info
        self._p = parent_op
        self._b = block
        self._nv_map: dict[str, Value] = {}

    @classmethod
    def define_function(
        cls, graph_info: GraphInfo, module_op: Operation
    ) -> "NodeImporter":
        cc = ContextCache(module_op.context)
        with module_op.context, Location.name(f"graph:{graph_info.graph_proto.name}"):
            body = module_op.regions[0].blocks[0]
            func_name = graph_info.graph_proto.name
            input_types = [
                cc.type_proto_to_type(inp.type) for inp in graph_info.input_map.values()
            ]
            output_types = [
                cc.type_proto_to_type(out.type)
                for out in graph_info.output_map.values()
            ]
            ftype = FunctionType.get(input_types, output_types)
            func_op = func_dialect.FuncOp(func_name, ftype, ip=InsertionPoint(body))
            block = func_op.add_entry_block(
                [Location.name(k) for k in graph_info.input_map.keys()]
            )
        imp = NodeImporter(graph_info, parent_op=func_op, block=block, context_cache=cc)
        for node_name, input_value in zip(graph_info.input_map.keys(), block.arguments):
            imp._nv_map[node_name] = input_value
        imp._populate_graph_attrs(func_op)
        return imp

    def _populate_graph_attrs(self, container_op: Operation):
        """Populates graph level meta attributes on the given container op."""
        m = self._gi.model_info.model_proto
        with container_op.context:
            i64_type = IntegerType.get_signed(64)
            default_opset_version = 0
            opset_versions: dict[str, IntegerAttr] = {}
            for opset_import in m.opset_import:
                if opset_import.domain:
                    opset_versions[opset_import.domain] = IntegerAttr.get(
                        i64_type, opset_import.version
                    )
                else:
                    default_opset_version = opset_import.version
            if default_opset_version:
                container_op.attributes[
                    "torch.onnx_meta.opset_version"
                ] = IntegerAttr.get(i64_type, default_opset_version)
            if opset_versions:
                container_op.attributes[
                    "torch.onnx_meta.opset_versions"
                ] = DictAttr.get(opset_versions)
            container_op.attributes["torch.onnx_meta.ir_version"] = IntegerAttr.get(
                IntegerType.get_signed(64), m.ir_version
            )
            container_op.attributes["torch.onnx_meta.producer_name"] = StringAttr.get(
                m.producer_name
            )
            container_op.attributes[
                "torch.onnx_meta.producer_version"
            ] = StringAttr.get(m.producer_version)

    def import_all(self):
        """Imports all nodes topologically."""
        # TODO: Consider pulling in initializers on demand since there can be so
        # much unused crap.
        for init in self._gi.initializer_map.values():
            self.import_initializer(init)

        self.get_none()
        for node in self._gi.graph_proto.node:
            self.import_node(node)

        outputs = []
        for output_name in self._gi.output_map.keys():
            try:
                outputs.append(self._nv_map[output_name])
            except KeyError:
                raise OnnxImportError(
                    f"Non topologically produced ONNX graph output '{output_name}'"
                )
        with InsertionPoint(self._b), Location.unknown():
            func_dialect.ReturnOp(outputs)

    def get_none(self):
        if '' in self._nv_map:
            return self._nv_map['']

        with InsertionPoint(self._b), Location.name("onnx_importer.none"):
            nne = Operation.create(
                name="torch.constant.none",
                results=[self._cc.get_none_type()],
                operands=[],
                attributes={},
            ).results[0]
            self._nv_map[''] = nne
            return nne

    def import_node(self, node: onnx.NodeProto):
        with InsertionPoint(self._b), Location.name(node.name):
            op_type = node.op_type
            # Handle special op types that materialize to non-op IR constructs.
            # Handlers return True if the op was handled, else this function
            # should process it as a general node.
            special_key = f"_handle_node_{op_type}"
            if hasattr(self, special_key):
                was_handled = getattr(self, special_key)(node)
                if was_handled:
                    return
            # General node import.
            input_values = []
            for input_name in node.input:
                try:
                    input_values.append(self._nv_map[input_name])
                except KeyError:
                    raise OnnxImportError(
                        f"Non topologically produced ONNX node input '{input_name}': {node}"
                    )

            output_names = list(node.output)
            output_types = [
                self._cc.type_proto_to_type(self._gi.find_type_proto_for_name(n))
                for n in output_names
            ]

            # TODO: Attributes.
            attrs = {
                "name": StringAttr.get(f"onnx.{op_type}"),
            }
            self.import_attributes(node.attribute, attrs)
            custom_op = Operation.create(
                name="torch.operator",
                results=output_types,
                operands=input_values,
                attributes=attrs,
            )
            for output_name, output_value in zip(output_names, custom_op.results):
                self._nv_map[output_name] = output_value

    def import_attributes(
        self, onnx_attrs: list[onnx.AttributeProto], attrs: dict[str, Attribute]
    ):
        for onnx_attr in onnx_attrs:
            attr_type = onnx_attr.type
            if attr_type not in ATTRIBUTE_TYPE_HANDLERS:
                raise OnnxImportError(
                    f"Unhandled ONNX attribute type code {attr_type}: {onnx_attr}"
                )
            handler = ATTRIBUTE_TYPE_HANDLERS[attr_type]
            if handler is None:
                # Active skip.
                continue
            elif handler is False:
                # Active error.
                raise OnnxImportError(
                    f"ONNX importer does not support generic node attribute type {attr_type}. "
                    f"This likely means that this is a special node which requires specific "
                    f"handling in the importer: {onnx_attr}"
                )
            result = handler(onnx_attr, self._cc)
            attrs[f"torch.onnx.{onnx_attr.name}"] = result

    def import_initializer(self, initializer: onnx.TensorProto, extern_name: str = None) -> Value:
        # If an explicitly specified name is given, use that; otherwise, pick
        # up the name from the tensor proto itself
        iname = extern_name if extern_name else initializer.name
        with InsertionPoint(self._b), Location.name(iname):
            value_attr = self._cc.tensor_proto_to_attr(initializer)
            vtensor_type = self._cc.tensor_proto_to_type(initializer)
            attrs = {
                "name": StringAttr.get(f"onnx.Constant"),
                "torch.onnx.value": value_attr,
            }
            literal_op = Operation.create(
                name="torch.operator",
                results=[vtensor_type],
                attributes=attrs,
            )
            self._nv_map[iname] = literal_op.result
        return literal_op.result

    def _get_immediate_tensor(self, name: str) -> np.array:
        try:
            initializer = self._gi.initializer_map[name]
        except KeyError:
            raise OnnxImportError(
                f"An immediate value for '{name}' was required but it is dynamically produced."
            )
        try:
            dtype = ELEM_TYPE_TO_NUMPY_DTYPE[initializer.data_type]
        except KeyError:
            raise OnnxImportError(
                f"Unknown ONNX tensor element type to numpy dtype mapping: {initializer.data_type}"
            )
        raw_data = initializer.raw_data
        if raw_data:
            return np.frombuffer(raw_data, dtype=dtype).reshape(tuple(initializer.dims))
        else:
            raise OnnxImportError(
                f"Unhandled ONNX TensorProto immediate data: {initializer}"
            )

    def _handle_node_Constant(self, node: onnx.NodeProto) -> bool:
        # Special case only for constants specified by value attribute (for now)
        value_proto = _get_attr(node, "value", False)
        if not value_proto:
            return False

        # Produce an initializer for the constant, so that it can be used in
        # combination with other ops, such as ConstantOfShape, requiring
        # a constant input
        assert value_proto.type == onnx.AttributeProto.AttributeType.TENSOR
        assert len(node.output) == 1
        const_name = node.output[0]
        self.import_initializer(value_proto.t, const_name)
        self._gi.initializer_map[const_name] = value_proto.t
        return True

class ContextCache:
    """Caches per-context lookups of various things."""

    __slots__ = [
        "_c",
        "_elem_type_map",
        "_vtensor_type_map",
    ]

    def __init__(self, context: Context):
        self._c = context
        self._elem_type_map: dict[int, IrType] = {}
        self._vtensor_type_map: dict[tuple[tuple[Optional[int]], IrType], IrType] = {}

    def tensor_element_type(self, elem_type: int) -> IrType:
        t = self._elem_type_map.get(elem_type)
        if t is None:
            try:
                with self._c:
                    t = ELEM_TYPE_TO_IR_TYPE_CB[elem_type]()
            except KeyError:
                raise OnnxImportError(f"Unknown ONNX tensor element type: {elem_type}")
            self._elem_type_map[elem_type] = t
        return t

    def get_none_type(self):
        return IrType.parse("!torch.none", context=self._c)

    def get_vtensor_type(
        self, dims: tuple[Optional[int]], element_type: IrType
    ) -> IrType:
        key = (dims, element_type)
        t = self._vtensor_type_map.get(key)
        if t is None:
            shape_asm = ",".join("?" if d is None else str(d) for d in dims)
            asm = f"!torch.vtensor<[{shape_asm}],{str(element_type)}>"
            try:
                t = IrType.parse(asm, context=self._c)
            except MLIRError as e:
                raise OnnxImportError(
                    f"Unparseable torch type (MLIR asm format bug?): {asm}"
                ) from e
            self._vtensor_type_map[key] = t
        return t

    def tensor_proto_to_type(self, tp: onnx.TensorProto) -> IrType:
        element_type = self.tensor_element_type(tp.data_type)
        return self.get_vtensor_type(tuple(tp.dims), element_type)

    def tensor_proto_to_builtin_type(self, tp: onnx.TensorProto) -> IrType:
        element_type = self.tensor_element_type(tp.data_type)
        # TODO: Fixme upstream: RankedTensorType.get should not require a location.
        with Location.unknown():
            return RankedTensorType.get(tuple(tp.dims), element_type)

    def type_proto_to_type(self, tp: onnx.TypeProto) -> IrType:
        if tp.tensor_type:
            tt = tp.tensor_type
            if not tt.shape:
                raise OnnxImportError(
                    f"Unsupported Tensor type without shape (run shape inference?): {tp}"
                )
            element_type = self.tensor_element_type(tt.elem_type)
            dims = tuple(
                (d.dim_value if not d.dim_param else None) for d in tt.shape.dim
            )
            return self.get_vtensor_type(dims, element_type)
        else:
            # TODO: Others if ever needed. Or we consider ourselves DNN-only.
            # See TypeProto: sequence_type, map_type, optional_type, sparse_tensor_type.
            raise OnnxImportError(f"Unsupported ONNX TypeProto: {tp}")

    def _sanitize_name(self, name):
        if not name.isidentifier():  
            name = "_" + name  
        return re.sub("[:/]", "_", name)  

    def tensor_proto_to_attr(self, tp: onnx.TensorProto) -> Attribute:
        tensor_type = self.tensor_proto_to_builtin_type(tp)
        if tp.HasField("raw_data"):
            # Conveniently, DenseResourceElementsAttr shares the raw data
            # format. We just give it maximum numeric alignment.
            resource = DenseResourceElementsAttr.get_from_buffer(
                tp.raw_data, self._sanitize_name(tp.name), tensor_type, alignment=8
            )
            return resource
        else:
            # We have to do a data type specific instantiation from proto fields.
            # Since this is typically used for small tensor constants, we instantiate
            # as a DenseElementsAttr.
            handler = ELEM_TYPE_INLINE_TENSOR_PROTO_CB.get(tp.data_type)
            if handler is None:
                raise OnnxImportError(f"Unhandled ONNX TensorProto data: {tp}")
            return handler(tp)


ELEM_TYPE_TO_IR_TYPE_CB = {
    onnx.TensorProto.DataType.FLOAT: lambda: F32Type.get(),
    onnx.TensorProto.DataType.UINT8: lambda: IntegerType.get_unsigned(8),
    onnx.TensorProto.DataType.INT8: lambda: IntegerType.get_signed(8),
    onnx.TensorProto.DataType.UINT16: lambda: IntegerType.get_unsigned(16),
    onnx.TensorProto.DataType.INT16: lambda: IntegerType.get_signed(16),
    onnx.TensorProto.DataType.INT32: lambda: IntegerType.get_signed(32),
    onnx.TensorProto.DataType.INT64: lambda: IntegerType.get_signed(64),
    onnx.TensorProto.DataType.BOOL: lambda: IntegerType.get_signless(1),
    onnx.TensorProto.DataType.FLOAT16: lambda: F16Type.get(),
    onnx.TensorProto.DataType.DOUBLE: lambda: F64Type.get(),
    onnx.TensorProto.DataType.UINT32: lambda: IntegerType.get_unsigned(32),
    onnx.TensorProto.DataType.UINT64: lambda: IntegerType.get_unsigned(64),
    onnx.TensorProto.DataType.COMPLEX64: lambda: ComplexType.get(F32Type.get()),
    onnx.TensorProto.DataType.COMPLEX128: lambda: ComplexType.get(F64Type.get()),
    onnx.TensorProto.DataType.BFLOAT16: lambda: BF16Type.get(),
    onnx.TensorProto.DataType.FLOAT8E4M3FN: lambda: Float8E4M3FNType.get(),
    onnx.TensorProto.DataType.FLOAT8E4M3FNUZ: lambda: Float8E5M2FNUZType.get(),
    onnx.TensorProto.DataType.FLOAT8E5M2: lambda: Float8E5M2Type.get(),
    onnx.TensorProto.DataType.FLOAT8E5M2FNUZ: lambda: Float8E5M2FNUZType.get(),
    # Ommitted: STRING,
}

ELEM_TYPE_SPLAT_TENSOR_PROTO_CB = {
    onnx.TensorProto.DataType.FLOAT: lambda tp, shape: DenseElementsAttr.get_splat(
        RankedTensorType.get(shape, F32Type.get()), FloatAttr.get_f32(tp.float_data[0])
    ),
    onnx.TensorProto.DataType.INT64: lambda tp, shape: DenseElementsAttr.get_splat(
        RankedTensorType.get(shape, IntegerType.get_signed(64)), IntegerAttr.get(
        IntegerType.get_signed(64), int.from_bytes(tp.raw_data, "little",
        signed=True) if tp.HasField("raw_data") else tp.int64_data[0])
    ),
    # TODO: All the rest from ELEM_TYPE_TO_IR_TYPE_CB
}

# Mapping of TensorProto.DataType to lambda TensorProto, returning a DenseElementsAttr
# of the builtin tensor type for cases where the tensor data is inlined as typed
# values instead of raw_data.
ELEM_TYPE_INLINE_TENSOR_PROTO_CB = {
    onnx.TensorProto.DataType.FLOAT: lambda tp: DenseElementsAttr.get(
        np.asarray(tp.float_data, dtype=np.float32).reshape(tp.dims), signless=False
    ),
    onnx.TensorProto.DataType.INT32: lambda tp: DenseElementsAttr.get(
        np.asarray(tp.int32_data, dtype=np.int32).reshape(tp.dims), signless=False
    ),
    onnx.TensorProto.DataType.INT64: lambda tp: DenseElementsAttr.get(
        np.asarray(tp.int64_data, dtype=np.int64).reshape(tp.dims), signless=False
    ),
    onnx.TensorProto.DataType.DOUBLE: lambda tp: DenseElementsAttr.get(
        np.asarray(tp.double_data, dtype=np.float64).reshape(tp.dims)
    ),
    onnx.TensorProto.DataType.UINT32: lambda tp: DenseElementsAttr.get(
        # Special case. See proto
        np.asarray(tp.uint64_data, dtype=np.uint32).reshape(tp.dims),
        signless=False,
    ),
    onnx.TensorProto.DataType.UINT64: lambda tp: DenseElementsAttr.get(
        np.asarray(tp.uint64_data, dtype=np.uint64).reshape(tp.dims), signless=False
    )
    # Intentionally unsupported: STRING
}

ELEM_TYPE_TO_NUMPY_DTYPE = {
    onnx.TensorProto.DataType.FLOAT: np.float32,
    onnx.TensorProto.DataType.UINT8: np.uint8,
    onnx.TensorProto.DataType.INT8: np.int8,
    onnx.TensorProto.DataType.UINT16: np.uint16,
    onnx.TensorProto.DataType.INT16: np.int16,
    onnx.TensorProto.DataType.INT32: np.int32,
    onnx.TensorProto.DataType.INT64: np.int64,
    onnx.TensorProto.DataType.BOOL: np.bool_,
    onnx.TensorProto.DataType.FLOAT16: np.float16,
    onnx.TensorProto.DataType.DOUBLE: np.float64,
    onnx.TensorProto.DataType.UINT32: np.uint32,
    onnx.TensorProto.DataType.UINT64: np.uint64,
    onnx.TensorProto.DataType.COMPLEX64: np.complex64,
    onnx.TensorProto.DataType.COMPLEX128: np.complex128,
    # onnx.TensorProto.DataType.BFLOAT16:
    # onnx.TensorProto.DataType.FLOAT8E4M3FN:
    # onnx.TensorProto.DataType.FLOAT8E4M3FNUZ:
    # onnx.TensorProto.DataType.FLOAT8E5M2:
    # onnx.TensorProto.DataType.FLOAT8E5M2FNUZ:
    # Ommitted: STRING,
}

# Mapping of AttributeType code to one of:
#   None: Ignore attribute and do not output to MLIR
#   False: Error if an attribute of this type is present
#   lambda a:AttributeProto, cc: ContextCache that returns an MLIR Attribute
ATTRIBUTE_TYPE_HANDLERS = {
    onnx.AttributeProto.AttributeType.UNDEFINED: False,
    onnx.AttributeProto.AttributeType.FLOAT: lambda a, cc: FloatAttr.get(
        F32Type.get(), a.f
    ),
    onnx.AttributeProto.AttributeType.INT: lambda a, cc: IntegerAttr.get(
        IntegerType.get_signed(64), a.i
    ),
    onnx.AttributeProto.AttributeType.STRING: lambda a, cc: StringAttr.get(a.s),
    onnx.AttributeProto.AttributeType.TENSOR: lambda a, cc: cc.tensor_proto_to_attr(
        a.t
    ),
    onnx.AttributeProto.AttributeType.GRAPH: False,
    onnx.AttributeProto.AttributeType.SPARSE_TENSOR: False,
    onnx.AttributeProto.AttributeType.TYPE_PROTO: False,
    onnx.AttributeProto.AttributeType.FLOATS: lambda a, cc: ArrayAttr.get(
        [FloatAttr.get(F32Type.get(), f) for f in a.floats]
    ),
    onnx.AttributeProto.AttributeType.INTS: lambda a, cc: ArrayAttr.get(
        [IntegerAttr.get(IntegerType.get_signed(64), i) for i in a.ints]
    ),
    onnx.AttributeProto.AttributeType.STRINGS: lambda a, cc: ArrayAttr.get(
        [StringAttr.get(s) for s in a.strings]
    ),
    onnx.AttributeProto.AttributeType.TENSORS: lambda a, cc: ArrayAttr.get(
        [cc.tensor_proto_to_attr(t) for t in a.tensors]
    ),
    onnx.AttributeProto.AttributeType.GRAPHS: False,
    onnx.AttributeProto.AttributeType.SPARSE_TENSORS: False,
    onnx.AttributeProto.AttributeType.TYPE_PROTOS: False,
}


def _get_attr(node: onnx.NodeProto, attr_name: str, is_required: bool = True) -> onnx.AttributeProto:
    for attr in node.attribute:
        if attr.name == attr_name:
            return attr
    if is_required:
        raise OnnxImportError(f"Required attribute {attr_name} not found in {node}")
    return None
