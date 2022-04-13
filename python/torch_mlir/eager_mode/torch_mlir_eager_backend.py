# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import abc
from dataclasses import dataclass
from typing import TypeVar, Tuple, Callable, List, Dict, Any

import torch

from torch_mlir._mlir_libs._mlir.ir import Module

# TODO: This might need to be an ABC too, such as
# to support finding the backend that created the tensor.
DeviceTensor = TypeVar("DeviceTensor")


@dataclass(frozen=True)
class TensorMetaData:
    """A small container for metadata necessary for satisfying the pytorch dispatcher and other code (pytorch or
    otherwise) that branches on these attributes.

    There is a lot of code in the PyTorch codebase that branches based on these attributes; the obvious ones here
    are dtype, device, and requires_grad (necessary for autograd itself). There is ample warning from PyTorch that,
    in principle, these should be as close as possible to true; see
    https://github.com/albanD/subclass_zoo/blob/1566e038f03cd89ab3cc37e670a44e3c2bbc1897/trivial_tensors.py#L90-L92

    The defaults (properties) simplify the api and seem to work after some testing but
    might malfunction in unexpected ways.
    # TODO: revisit these assumptions
    """

    size: Tuple[int]
    dtype: torch.dtype
    requires_grad: bool

    strides: Tuple[int]
    storage_offset: int = 0
    layout: torch.layout = torch.strided
    device: torch.device = torch.device("cpu")

    def __init__(
        self,
        size,
        dtype,
        requires_grad,
        strides=None,
        storage_offset=None,
        layout=None,
        device=None,
    ):
        super().__init__()
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "requires_grad", requires_grad)

        object.__setattr__(
            self, "strides", strides if strides is not None else len(size) * [0]
        )
        object.__setattr__(
            self, "storage_offset", storage_offset if storage_offset is not None else 0
        )
        object.__setattr__(
            self, "layout", layout if layout is not None else torch.strided
        )
        object.__setattr__(
            self, "device", device if device is not None else torch.device("cpu")
        )


class TorchMLIREagerBackend(abc.ABC):
    @abc.abstractmethod
    def compile(
        self, module: Module
    ) -> Callable[[List[DeviceTensor]], List[DeviceTensor]]:
        raise NotImplementedError

    @abc.abstractmethod
    def transfer_from_torch_to_device(self, tensor: torch.Tensor) -> DeviceTensor:
        """Unwrap the backend representation in order to build a torch.Tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_torch_metadata(
        self, tensor: DeviceTensor, kwargs: Dict[str, Any]
    ) -> TensorMetaData:
        """Parse relevant tensor metadata from backend device array (e.g., shape, stride, layout) in order to build
        wrapper tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def transfer_from_device_to_torch(self, tensor: DeviceTensor) -> torch.Tensor:
        """If compilation fails for some reason then device specific representations need to be munged into a
        torch.Tensor representation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def copy_into(self, dst: DeviceTensor, src: DeviceTensor):
        """This method is needed for things like handling aliased arguments."""
        raise NotImplementedError
