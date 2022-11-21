# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
import re
import traceback
import warnings
from typing import Any

import torch
from torch.utils._pytree import tree_map

from torch_mlir.eager_mode.ir_building import build_mlir_module
from torch_mlir.eager_mode.torch_mlir_dispatch import (
    UnsupportedByTorchMlirEagerMode,
    normalize_args_kwargs,
    check_get_aliased_arg,
)
from torch_mlir.eager_mode import EAGER_MODE_DEBUG
from torch_mlir_e2e_test.eager_backends.refbackend import EagerModeRefBackend


@contextlib.contextmanager
def no_dispatch():
    """Prevent infinite recursion in case accidentally calling a tensor method on a TorchMLIRTensor within
    __torch_dispatch__."""

    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


backend = EagerModeRefBackend()

UNSUPPORTED_OPS = re.compile(
    "|".join([
        # We don't handle detach as it only pertains to autograd graph construction, which is handled by pytorch.
        "detach",
        # We don't handle _local_scalar_dense because it's just a way to unwrap a tensor that wraps a number.
        "_local_scalar_dense",
        # https://github.com/llvm/torch-mlir/issues/878
        "_unsafe_view",
        "view",
    ])
)


class TorchMLIRTensor(torch.Tensor):
    """This class serves the role abstract class with common functionality for dispatching through Torch-MLIR instead of aten.

    It defers device specific behavior to device specific implementations. The deriving classes use the
    make_bare_wrapper_subclass convenience method, adjacent here, and override __torch_dispatch__ in order to dispatch
    through Torch-MLIR instead of aten. Backends are free to choose whatever representation of the buffers (i.e., `elem`)
    and are expected to provide conversion mechanisms between their representation and torch.Tensor.

    Here we only verify that inputs abide by current supported features of Torch-MLIR (contiguous memory and
    strided tensor layout) and build the mlir module. Importantly, we also recover from any malfunctions in the
    deriving classes and dispatch back to conventional PyTorch.

    More documentation on how the __torch_dispatch__ pattern works can be found in this forum post
    https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
    and this RFC
    https://github.com/pytorch/rfcs/blob/master/RFC-0001-torch-function-for-methods.md#process-followed-during-a-functionmethod-call
    and this repo with many examples
    https://github.com/albanD/subclass_zoo
    """

    elem: Any

    __slots__ = ["elem"]

    def __new__(cls, elem, **kwargs):
        """Wrap elem (which could be a torch.Tensor or otherwise) in a torch.Tensor subclass.

        Critically, this method needs to parse relevant metadata from the device representation
        (such as shape, striding, dtype, etc.) and translate it into torch conventions.

        Deriving classes must provide a way to construct themselves from either their device specific representation
        or torch.Tensor; the latter is to handle the case that dispatch to PyTorch to recover from an error.
        """
        if kwargs.get("constructing_from_device_tensor", False):
            tensor_meta_data = backend.get_torch_metadata(elem, kwargs)
            r = make_bare_wrapper_subclass(
                cls=cls,
                size=tensor_meta_data.size,
                strides=tensor_meta_data.strides,
                storage_offset=tensor_meta_data.storage_offset,
                dtype=tensor_meta_data.dtype,
                layout=tensor_meta_data.layout,
                device=tensor_meta_data.device,
                requires_grad=tensor_meta_data.requires_grad,
            )
            r.elem = elem
        elif isinstance(elem, torch.nn.Parameter):
            r = make_wrapper_subclass_from_torch_tensor(cls, elem.data, **kwargs)
            r.elem = backend.transfer_from_torch_to_device(elem.detach().data)
        elif isinstance(elem, torch.Tensor):
            r = make_wrapper_subclass_from_torch_tensor(cls, elem, **kwargs)
            r.elem = backend.transfer_from_torch_to_device(elem)
        # This branch handles the case when a python scalar is passed to some op
        # or is returned from some aten op, such as _local_scalar_dense.
        elif isinstance(elem, (int, float, bool)):
            return elem
        else:
            raise ValueError(f"Unknown element type: {type(elem)}")

        return r

    def __repr__(self):
        if self.grad_fn:
            return f"TorchMLIRTensor({self.elem}, backend={backend.__class__.__name__}, grad_fn={self.grad_fn})"
        else:
            return f"TorchMLIRTensor({self.elem}, backend={backend.__class__.__name__})"

    @classmethod
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):
        requires_grad = check_requires_grad(*args, **kwargs)
        try:
            with no_dispatch():
                if hasattr(func, "op_name"):
                    op_name = func.op_name
                elif hasattr(func, "__name__"):
                    # Handle builtin_function_or_method.
                    op_name = func.__name__
                else:
                    raise RuntimeError(f"op {func} has no name")

                requires_grad = requires_grad and "view" not in op_name

                if UNSUPPORTED_OPS.match(op_name):
                    raise UnsupportedByTorchMlirEagerMode(op_name)

                if not hasattr(func, "_schema"):
                    raise RuntimeError(f"op {func} has no schema.")

                normalized_kwargs = normalize_args_kwargs(func, args, kwargs)

                if "layout" in normalized_kwargs and normalized_kwargs[
                    "layout"
                ] not in {0, None}:
                    raise UnsupportedByTorchMlirEagerMode(
                        f"{normalized_kwargs['layout']} layout not supported."
                    )
                if "memory_format" in normalized_kwargs and normalized_kwargs[
                    "memory_format"
                ] not in {0, None}:
                    raise UnsupportedByTorchMlirEagerMode(
                        f"{normalized_kwargs['memory_format']} memory format not supported."
                    )
                eager_module = build_mlir_module(func, normalized_kwargs)
            device_tensor_args = [
                kwarg.elem
                for _, kwarg in normalized_kwargs.items()
                if isinstance(kwarg, cls)
            ]
            assert len(eager_module.body.operations[0].arguments) == len(
                device_tensor_args
            ), "Number of parameters and number of arguments differs."
            op_mlir_backend_callable = backend.compile(eager_module)
            out = op_mlir_backend_callable(*device_tensor_args)
            out = tree_map(
                lambda x: cls(
                    x, requires_grad=requires_grad, constructing_from_device_tensor=True
                ),
                out,
            )
        except Exception as e:
            if EAGER_MODE_DEBUG:
                warnings.warn(traceback.format_exc())
                if isinstance(e, UnsupportedByTorchMlirEagerMode):
                    warnings.warn(
                        f"Couldn't use TorchMLIR eager because current incompatibility: *{str(e)}*; running through PyTorch eager."
                    )
                else:
                    warnings.warn(
                        f"Couldn't use TorchMLIR eager because of error: *{str(e)}*; "
                        f"running through PyTorch eager. Please file an issue at https://github.com/llvm/torch-mlir/issues"
                    )

            with no_dispatch():
                unwrapped_args = tree_map(cls.unwrap, args)
                unwrapped_kwargs = tree_map(cls.unwrap, kwargs)
                out = func(*unwrapped_args, **unwrapped_kwargs)

            out = tree_map(lambda x: cls(x, requires_grad=requires_grad), out)

        maybe_aliased_arg_name = check_get_aliased_arg(func)
        if maybe_aliased_arg_name is not None:
            backend.copy_into(normalized_kwargs[maybe_aliased_arg_name].elem, out.elem)

        return out

    @classmethod
    def unwrap(cls, e):
        """Unwrap the TorchMLIRTensor representation in order to access the actual device specific representation."""
        if isinstance(e, cls):
            return backend.transfer_from_device_to_torch(e.elem)
        return e


def check_requires_grad(*args, **kwargs):
    requires_grad = False

    def check_grad(e):
        nonlocal requires_grad
        if isinstance(e, TorchMLIRTensor):
            requires_grad |= e.requires_grad

    tree_map(check_grad, args)
    tree_map(check_grad, kwargs)

    return requires_grad


def make_wrapper_subclass_from_torch_tensor(cls, elem, **kwargs):
    """Convenience method that parse out relevant metadata from a torch.Tensor, in order to produce
    a wrapper subclass.

    NB: this convenience method does not set that `elem` attribute of the subclass, as that is the responsibility
    of the device specific implementation.
    """
    r = make_bare_wrapper_subclass(
        cls=cls,
        size=elem.size(),
        strides=elem.stride(),
        storage_offset=elem.storage_offset(),
        dtype=elem.dtype,
        layout=elem.layout,
        device=elem.device,
        # Only float tensors can have gradients.
        requires_grad=elem.dtype in {torch.float, torch.float32, torch.float64}
        and (kwargs.get("requires_grad", False) or elem.requires_grad),
    )
    return r


def make_bare_wrapper_subclass(
    *, cls, size, strides, storage_offset, dtype, layout, device, requires_grad
):
    """Convenience method that builds a wrapper subclass.

    NB: this convenience method does not set that `elem` attribute of the subclass, as that is the responsibility
    of the device specific implementation.
    """
    return torch.Tensor._make_wrapper_subclass(
        cls,
        size,
        strides=strides,
        storage_offset=storage_offset,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
