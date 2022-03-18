# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Translator from torch.jit.ScriptFunction to MLIR.


The following defines a set of classes for converting types used by Python and PyTorch into MLIR types from the
`torch` dialect.

The expected use of this module is to create an instance of one of the classes below, and then calling the
`to_mlir` method to generate the MLIR representation of the type.

Information about what types are supported by each class can be found in docstrings of each of the classes.

In addition this module defines a function that take a torch.jit.ScriptFunction and converts it into an MLIR module.

The expected use for this module is to use the function
`build_module(jit_function: torch.jit.ScriptFunction annotation: Annotation) -> ir.Module`
to convert the TorchScript function into MLIR using the `torch` dialect.
"""

import abc
from typing import Any, Optional, Iterable
from typing import Union

import torch
from torch.jit import ScriptFunction

from torch_mlir import ir
from torch_mlir.dialects.func import FuncOp
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder


class TorchMlirType(abc.ABC):
    """
    A `TorchMlirType` is an object that produces MLIR
    types in the `torch` dialect. The only requirement
    for a class to be a subclass of `TorchMlirType`  is
    to define a `to_mlir(self, ir.Context) -> ir.Type`.
    Each class is allowed to have different types of
    __init__ methods depending on the information they
    require to produce the given MLIR representation.
    """

    @abc.abstractmethod
    def to_mlir(self, context: ir.Context) -> ir.Type:
        pass


class TorchTensorTypeError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


class TorchTensorType(TorchMlirType):
    """
    This class is used to generate types of the form
    !torch.tensor and !torch.vtensor<SHAPE, DTYPE>,
    where SHAPE is a list representing the shape of the tensor,
    and DTYPE is an MLIR data type.
    """

    def __init__(
        self,
        *,
        shape: Optional[Iterable[Optional[int]]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.shape = shape
        self.dtype = dtype

        if dtype is None and shape is not None:
            err = "If shape is specified, dtype must also be specified"
            raise TorchTensorTypeError(err)

    def __str__(self):
        return f"Torch Tensor (shape={self.shape}, dtype={self.dtype})"

    def to_mlir(self, context: ir.Context) -> ir.Type:
        if self.dtype is None:
            return ir.Type.parse("!torch.tensor", context=context)

        shape_asm = self._shape_to_mlir_asm()
        dtype_asm = self._dtype_to_mlir_asm()
        return ir.Type.parse(
            f"!torch.vtensor<{shape_asm},{dtype_asm}>", context=context
        )

    def _shape_to_mlir_asm(self) -> str:
        if self.shape is None:
            return "*"

        str_sizes = map(lambda x: "?" if x is None else str(x), self.shape)
        return f'[{",".join(str_sizes)}]'

    def _dtype_to_mlir_asm(self) -> str:
        if self.dtype in [torch.float64]:
            return "f64"
        if self.dtype in [torch.float, torch.float32]:
            return "f32"
        if self.dtype in [torch.int, torch.int32]:
            return "si32"
        if self.dtype in [torch.int64]:
            return "si64"
        if self.dtype in [torch.bool]:
            return "i1"

        raise NotImplementedError(f"Unsupported dtype: {self.dtype}")


class TorchNnModuleType(TorchMlirType):
    """This class is used to generate types for `!torch.nn.Module`s."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def __str__(self):
        return "torch.nn.Module"

    def to_mlir(self, context: ir.Context) -> ir.Type:
        return ir.Type.parse(f'!torch.nn.Module<"{self.module_name}">', context=context)


class PythonType(TorchMlirType):
    """
    This class is used to convert regular Python types
    into their corresponding `torch` dialect representation.
    The list of supported types can be found in the dictionary
    `_type_to_asm_dict`.
    """

    _type_to_asm_dict = {
        bool: "!torch.bool",
        int: "!torch.int",
        type(None): "!torch.none",
    }

    def __init__(self, type_: Any):
        self.type_ = type_

    def __str__(self):
        return str(self.type_)

    def to_mlir(self, context: ir.Context) -> ir.Type:
        asm = self._type_to_asm_dict.get(self.type_)
        if asm is None:
            raise NotImplementedError(f"Unsupported type: {self.type_}")
        return ir.Type.parse(asm, context=context)


# TODO: This functionality should be incorporated into ModuleBuilder.import_function.
class Annotation:
    def __init__(self, types: Iterable[Union[TorchTensorType, type]]):
        self.types = list(
            map(lambda t: PythonType(t) if isinstance(t, type) else t, types)
        )

    def __str__(self):
        result = f"Annotation instance with {len(self.types)} types\n"
        for e, type_ in enumerate(self.types):
            result += f"    Type of argument {e + 1}: {str(type_)}\n"
        return result

    def __iter__(self):
        return iter(self.types)


class AnnotationConverter:
    @staticmethod
    def to_mlir_array_attr(annotation: Annotation, context: ir.Context) -> ir.ArrayAttr:
        dict_attrs = []
        for type_ in annotation:
            if not isinstance(type_, TorchTensorType):
                dict_attrs.append(ir.DictAttr.get({}, context=context))
                continue

            ir_type = type_.to_mlir(context)
            with context:
                type_attr = ir.TypeAttr.get(ir_type)
                dict_attr = ir.DictAttr.get({"torch.type_bound": type_attr})
                dict_attrs.append(dict_attr)

        return ir.ArrayAttr.get(dict_attrs, context=context)


def get_func_op_with_name(module: ir.Module, name: str) -> Optional[FuncOp]:
    with module.context:
        name_attr = ir.StringAttr.get(name)
    for op in module.body.operations:
        if isinstance(op, FuncOp) and op.name == name_attr:
            # Add name of torch op as debug_module_name so that
            # run_pipeline_with_repro_report can use it.
            module.operation.attributes["torch.debug_module_name"] = name_attr
            return op

    return None


def build_module(jit_function: ScriptFunction, annotations) -> ir.Module:
    """Translate input function into an MLIR module in the `torch` dialect.

    Parameters
    ----------
    jit_function: ScriptFunction
        Function in TorchScript IR to turn into MLIR.
    annotation: Annotation
        Annotation object representing the types of
        the operands of `jit_function`.

    Returns
    -------
    ir.Module
        Translation of the input module into an MLIR module
    """
    mb = ModuleBuilder()
    mb.import_function(jit_function)

    func_op = get_func_op_with_name(mb.module, jit_function.name)
    assert (
        func_op is not None
    ), "Unable to find FuncOp in new module. Make sure function was imported correctly into ModuleBuilder"

    func_annotation = Annotation(annotations)
    arg_attrs = AnnotationConverter.to_mlir_array_attr(func_annotation, mb.context)
    func_op.attributes["arg_attrs"] = arg_attrs

    return mb.module
