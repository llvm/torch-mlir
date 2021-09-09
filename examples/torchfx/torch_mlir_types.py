"""
-*- Python -*-
This file is licensed under a pytorch-style license
See frontends/pytorch/LICENSE for license information.

The following defines a set of classes for converting
types used by Python and PyTorch into MLIR types from the
`torch` dialect.

The expected use of this module is to create an instance
of one of the classes below, and then calling the
`to_mlir` method to generate the MLIR representation
of the type.

Information about what types are supported by each class
can be found in docstrings of each of the classes.
"""

# pylint: disable=no-member, no-name-in-module, invalid-name, missing-function-docstring, fixme

import abc
from typing import Any, Optional, Iterable

import torch
from torch_mlir import ir

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
    def __init__(self, *, shape: Optional[Iterable[Optional[int]]] = None,
                 dtype: Optional[torch.dtype] = None):
        self.shape = shape
        self.dtype = dtype

        if dtype is None and shape is not None:
            err = "If shape is specified, dtype must also be specified"
            raise TorchTensorTypeError(err)

    def to_mlir(self, context: ir.Context) -> ir.Type:
        if self.dtype is None:
            return ir.Type.parse('!torch.tensor', context=context)

        shape_asm = self._shape_to_mlir_asm()
        dtype_asm = self._dtype_to_mlir_asm()
        return ir.Type.parse(f'!torch.vtensor<{shape_asm},{dtype_asm}>',
                             context=context)

    def _shape_to_mlir_asm(self) -> str:
        if self.shape is None:
            return '*'

        str_sizes = map(lambda x: '?' if x is None else str(x), self.shape)
        return f'[{",".join(str_sizes)}]'

    def _dtype_to_mlir_asm(self) -> str:
        if self.dtype in [torch.float, torch.float32]:
            return 'f32'

        raise NotImplementedError(f'Unsupported dtype: {self.dtype}')


class TorchNnModuleType(TorchMlirType):
    """This class is used to generate types for `!torch.nn.Module`s."""
    def __init__(self, module_name: str):
        self.module_name = module_name

    def to_mlir(self, context: ir.Context) -> ir.Type:
        return ir.Type.parse(f'!torch.nn.Module<"{self.module_name}">',
                             context=context)


class PythonType(TorchMlirType):
    """
    This class is used to convert regular Python types
    into their corresponding `torch` dialect representation.
    The list of supported types can be found in the dictionary
    `_type_to_asm_dict`.
    """
    _type_to_asm_dict = {
        bool: '!torch.bool',
        int: '!torch.int',
        type(None): '!torch.none',
    }

    def __init__(self, type_: Any):
        self.type_ = type_

    def to_mlir(self, context: ir.Context) -> ir.Type:
        asm = self._type_to_asm_dict.get(self.type_)
        if asm is None:
            raise NotImplementedError(f'Unsupported type: {self.type_}')
        return ir.Type.parse(asm, context=context)
