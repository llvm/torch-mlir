# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import warnings

import torch
from torch.utils._pytree import tree_map

from torch_mlir.eager_mode.torch_mlir_dispatch import (
    try_torch_mlir_eager,
    UnsupportedByTorchMlirEagerMode,
)
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class TorchMLIRTensor(torch.Tensor):
    """Wrap torch.Tensor in order to dispatch through torch-mlir instead of aten.

    This class uses the _make_wrapper_subclass pattern to override __torch_dispatch__
    in order to dispatch through torch-mlir instead of aten. Here we basically only unwrap and wrap
    torch.Tensors. Most of the heavy lifting is done in the adjacent torch_mlir_dispatch module.

    More documentation on how this pattern works can be found in this forum post
    https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
    and this RFC
    https://github.com/pytorch/rfcs/blob/master/RFC-0001-torch-function-for-methods.md#process-followed-during-a-functionmethod-call
    and this repo with many examples
    https://github.com/albanD/subclass_zoo
    """

    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            # Only float tensors can have gradients.
            requires_grad=elem.dtype in {torch.float, torch.float32, torch.float64}
            and (kwargs.get("requires_grad", False) or elem.requires_grad),
        )
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"TorchMLIRTensor({self.elem}, grad_fn={self.grad_fn})"
        else:
            return f"TorchMLIRTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):
        requires_grad = False

        def check_grad(e):
            nonlocal requires_grad
            if isinstance(e, TorchMLIRTensor):
                requires_grad |= e.requires_grad

        tree_map(check_grad, args)
        tree_map(check_grad, kwargs)

        def unwrap(e):
            if isinstance(e, TorchMLIRTensor):
                return e.elem
            if isinstance(e, torch.nn.Parameter):
                return e.detach()
            return e

        def wrap(e):
            nonlocal requires_grad
            return (
                TorchMLIRTensor(e, requires_grad=requires_grad)
                if isinstance(e, torch.Tensor)
                else e
            )

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        try:
            out = try_torch_mlir_eager(
                func,
                unwrapped_args,
                unwrapped_kwargs,
                backend=refbackend.RefBackendLinalgOnTensorsBackend(),
            )
            if isinstance(out, tuple):
                out = [torch.from_numpy(o) for o in out]
            else:
                out = torch.from_numpy(out)
            return tree_map(wrap, out)
        except Exception as e:
            if isinstance(e, UnsupportedByTorchMlirEagerMode):
                warnings.warn(
                    f"Couldn't use TorchMLIR eager because current incompatibility: *{str(e)}*; running through PyTorch eager."
                )
            else:
                warnings.warn(
                    f"Couldn't use TorchMLIR eager because of error: *{str(e)}*; "
                    f"running through PyTorch eager. Please file an issue at https://github.com/llvm/torch-mlir/issues"
                )
            return tree_map(wrap, func(*unwrapped_args, **unwrapped_kwargs))
