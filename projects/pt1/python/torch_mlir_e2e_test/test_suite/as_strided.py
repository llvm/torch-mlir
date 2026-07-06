# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.annotations import annotate_args, export
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case


class AtenAsStridedAfterAliasDetachModule(torch.nn.Module):
    @export
    @annotate_args([None, ([3, 2, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice(x, 0, 1, 3, 1)
        view = torch.ops.aten.alias(view)
        view = torch.ops.aten.detach(view)
        return torch.ops.aten.as_strided(
            view, size=(2, 2), stride=(8, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterAliasDetachModule())
def AtenAsStridedAfterAliasDetachModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))


class AtenAsStridedAfterBroadcastToModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.broadcast_to(x, [2, 4])
        return torch.ops.aten.as_strided(
            view, size=(2, 3), stride=(0, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterBroadcastToModule())
def AtenAsStridedAfterBroadcastToModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(4, dtype=torch.float32).reshape(1, 4))


class AtenAsStridedAfterChainedViewsModule(torch.nn.Module):
    @export
    @annotate_args([None, ([3, 2, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice(x, 0, 1, 3, 1)
        view = torch.ops.aten.permute(view, [1, 0, 2])
        return torch.ops.aten.as_strided(view, size=(2, 2), stride=(12, 1))


@register_test_case(module_factory=lambda: AtenAsStridedAfterChainedViewsModule())
def AtenAsStridedAfterChainedViewsModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))


class AtenAsStridedAfterDiagonalModule(torch.nn.Module):
    @export
    @annotate_args([None, ([4, 5], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.diagonal(x, offset=0, dim1=0, dim2=1)
        return torch.ops.aten.as_strided(view, size=(3,), stride=(6,), storage_offset=0)


@register_test_case(module_factory=lambda: AtenAsStridedAfterDiagonalModule())
def AtenAsStridedAfterDiagonalModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(20, dtype=torch.float32).reshape(4, 5))


class AtenAsStridedAfterExpandAsModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 4], torch.float32, True)])
    def forward(self, x):
        other = torch.empty(2, 4, dtype=x.dtype, device=x.device)
        view = torch.ops.aten.expand_as(x, other)
        return torch.ops.aten.as_strided(
            view, size=(2, 3), stride=(0, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterExpandAsModule())
def AtenAsStridedAfterExpandAsModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(4, dtype=torch.float32).reshape(1, 4))


class AtenAsStridedAfterExpandModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.expand(x, [2, 4])
        return torch.ops.aten.as_strided(
            view, size=(2, 3), stride=(0, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterExpandModule())
def AtenAsStridedAfterExpandModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(4, dtype=torch.float32).reshape(1, 4))


class AtenAsStridedAfterExpandLeadingSingletonSliceModule(torch.nn.Module):
    @export
    @annotate_args([None, ([8], torch.float32, True)])
    def forward(self, x):
        base = torch.ops.aten.as_strided(x, size=(4,), stride=(1,), storage_offset=0)
        view = torch.ops.aten.expand(base, [1, 4])
        view = torch.ops.aten.slice(view, 0, 1, 1, 1)
        return torch.ops.aten.as_strided(view, size=(1,), stride=(1,))


@register_test_case(
    module_factory=lambda: AtenAsStridedAfterExpandLeadingSingletonSliceModule()
)
def AtenAsStridedAfterExpandLeadingSingletonSliceModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(8, dtype=torch.float32))


class AtenAsStridedAfterMovedimModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.movedim(x, source=1, destination=2)
        return torch.ops.aten.as_strided(
            view, size=(2, 4, 2), stride=(12, 1, 4), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterMovedimModule())
def AtenAsStridedAfterMovedimModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterNarrowModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 4, 5], torch.float32, True)])
    def forward(self, x):
        view = torch.narrow(x, dim=1, start=1, length=2)
        return torch.ops.aten.as_strided(view, size=(2, 3), stride=(20, 1))


@register_test_case(module_factory=lambda: AtenAsStridedAfterNarrowModule())
def AtenAsStridedAfterNarrowModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(40, dtype=torch.float32).reshape(2, 4, 5))


class AtenAsStridedAfterNestedAsStridedExplicitOffsetModule(torch.nn.Module):
    @export
    @annotate_args([None, ([10], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.as_strided(x, size=(2,), stride=(1,), storage_offset=2)
        return torch.ops.aten.as_strided(view, size=(2,), stride=(1,), storage_offset=0)


@register_test_case(
    module_factory=lambda: AtenAsStridedAfterNestedAsStridedExplicitOffsetModule()
)
def AtenAsStridedAfterNestedAsStridedExplicitOffsetModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(10, dtype=torch.float32))


class AtenAsStridedAfterNumpyTModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.numpy_T(x)
        return torch.ops.aten.as_strided(
            view, size=(4, 3, 1), stride=(1, 4, 12), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterNumpyTModule())
def AtenAsStridedAfterNumpyTModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterPermuteModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.permute(x, [0, 2, 1, 3])
        return torch.ops.aten.as_strided(
            view, size=(1, 3, 2, 2), stride=(24, 4, 12, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterPermuteModule())
def AtenAsStridedAfterPermuteModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4))


class AtenAsStridedAfterReshapeAliasModule(torch.nn.Module):
    @export
    @annotate_args([None, ([24], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten._reshape_alias(x, (4, 6), (1, 4))
        return torch.ops.aten.as_strided(view, size=(2, 2), stride=(1, 4))


@register_test_case(module_factory=lambda: AtenAsStridedAfterReshapeAliasModule())
def AtenAsStridedAfterReshapeAliasModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32))


class AtenAsStridedAfterReshapeFlattenModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.reshape(x, (2, 12))
        view = torch.flatten(view)
        return torch.ops.aten.as_strided(
            view, size=(3, 4), stride=(4, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterReshapeFlattenModule())
def AtenAsStridedAfterReshapeFlattenModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterSelectModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.select(x, dim=1, index=1)
        return torch.ops.aten.as_strided(view, size=(2, 2), stride=(12, 1))


@register_test_case(module_factory=lambda: AtenAsStridedAfterSelectModule())
def AtenAsStridedAfterSelectModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterSliceModule(torch.nn.Module):
    @export
    @annotate_args([None, ([4, 5, 6], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice(x, 0, 1, 4, 1)
        return torch.ops.aten.as_strided(view, size=(2, 3), stride=(6, 1))


@register_test_case(module_factory=lambda: AtenAsStridedAfterSliceModule())
def AtenAsStridedAfterSliceModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(120, dtype=torch.float32).reshape(4, 5, 6))


class AtenAsStridedAfterSliceWithExplicitOffsetModule(torch.nn.Module):
    @export
    @annotate_args([None, ([4, 5, 6], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice(x, 0, 1, 4, 1)
        return torch.ops.aten.as_strided(
            view, size=(2, 3), stride=(6, 1), storage_offset=0
        )


@register_test_case(
    module_factory=lambda: AtenAsStridedAfterSliceWithExplicitOffsetModule()
)
def AtenAsStridedAfterSliceWithExplicitOffsetModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(120, dtype=torch.float32).reshape(4, 5, 6))


class AtenAsStridedAfterSqueezeUnsqueezeModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 2, 1, 3], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.squeeze(x)
        view = torch.ops.aten.unsqueeze(view, 1)
        return torch.ops.aten.as_strided(
            view, size=(2, 1, 3), stride=(3, 0, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterSqueezeUnsqueezeModule())
def AtenAsStridedAfterSqueezeUnsqueezeModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3))


class AtenAsStridedAfterTModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.t(x)
        return torch.ops.aten.as_strided(
            view, size=(3, 2), stride=(1, 3), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterTModule())
def AtenAsStridedAfterTModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(6, dtype=torch.float32).reshape(2, 3))


class AtenAsStridedAfterTransposeModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.transpose(x, 1, 2)
        return torch.ops.aten.as_strided(
            view, size=(2, 3, 2), stride=(12, 4, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterTransposeModule())
def AtenAsStridedAfterTransposeModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterUnflattenModule(torch.nn.Module):
    @export
    @annotate_args([None, ([24], torch.float32, True)])
    def forward(self, x):
        view = torch.unflatten(x, 0, (4, 6))
        return torch.ops.aten.as_strided(
            view, size=(2, 3), stride=(6, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterUnflattenModule())
def AtenAsStridedAfterUnflattenModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32))


class AtenAsStridedAfterUnfoldModule(torch.nn.Module):
    @export
    @annotate_args([None, ([10], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.unfold(x, dimension=0, size=3, step=2)
        return torch.ops.aten.as_strided(
            view, size=(2, 2), stride=(2, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterUnfoldModule())
def AtenAsStridedAfterUnfoldModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(10, dtype=torch.float32))


class AtenAsStridedAfterUnsafeViewModule(torch.nn.Module):
    @export
    @annotate_args([None, ([3, 2, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice(x, 0, 1, 3, 1)
        view = torch.ops.aten._unsafe_view(view, [4, 4])
        return torch.ops.aten.as_strided(
            view, size=(2, 2), stride=(8, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterUnsafeViewModule())
def AtenAsStridedAfterUnsafeViewModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))


class AtenAsStridedAfterViewModule(torch.nn.Module):
    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.view(x, (4, 6))
        return torch.ops.aten.as_strided(
            view, size=(3, 4), stride=(6, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterViewModule())
def AtenAsStridedAfterViewModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


class AtenAsStridedAfterSliceStepModule(torch.nn.Module):
    @export
    @annotate_args([None, ([3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.slice.Tensor(x, 1, 0, 4, 2)
        return torch.ops.aten.as_strided.default(view, size=(3, 2), stride=(4, 2))


@register_test_case(module_factory=lambda: AtenAsStridedAfterSliceStepModule())
def AtenAsStridedAfterSliceStepModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(12, dtype=torch.float32).reshape(3, 4))


class AtenAsStridedAfterContiguousModule(torch.nn.Module):
    @export
    @annotate_args([None, ([3, 4], torch.float32, True)])
    def forward(self, x):
        view = torch.ops.aten.transpose.int(x, 0, 1)
        view = torch.ops.aten.contiguous.default(view)
        return torch.ops.aten.as_strided.default(
            view, size=(2, 2), stride=(3, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterContiguousModule())
def AtenAsStridedAfterContiguousModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(12, dtype=torch.float32).reshape(3, 4))


class AtenAsStridedChannelsLastInputModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 3, 4, 5], torch.float32, True)])
    def forward(self, x):
        x = torch.ops.aten.contiguous.default(x, memory_format=torch.channels_last)
        return torch.ops.aten.as_strided.default(
            x, size=(1, 4, 5, 3), stride=(60, 15, 3, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedChannelsLastInputModule())
def AtenAsStridedChannelsLastInputModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5))


class AtenAsStridedChannelsLastParameterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
        weight = weight.contiguous(memory_format=torch.channels_last)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    @export
    @annotate_args([None])
    def forward(self):
        return torch.ops.aten.as_strided.default(
            self.weight, size=(1, 4, 5, 3), stride=(60, 15, 3, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedChannelsLastParameterModule())
def AtenAsStridedChannelsLastParameterModule_basic(module, tu: TestUtils):
    module.forward()


class AtenAsStridedAfterToChannelsLastModule(torch.nn.Module):
    @export
    @annotate_args([None, ([1, 3, 4, 5], torch.float32, True)])
    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        return torch.ops.aten.as_strided.default(
            x, size=(1, 4, 5, 3), stride=(60, 15, 3, 1), storage_offset=0
        )


@register_test_case(module_factory=lambda: AtenAsStridedAfterToChannelsLastModule())
def AtenAsStridedAfterToChannelsLastModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5))
