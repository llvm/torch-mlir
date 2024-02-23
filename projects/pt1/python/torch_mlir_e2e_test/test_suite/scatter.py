# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class IndexPutImpl1DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=False,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl1DFloatNonAccumulateModule())
def IndexPutImpl1DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(100), tu.randint(250, high=100), tu.rand(250))


class IndexPutImpl2DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=False,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl2DFloatNonAccumulateModule())
def IndexPutImpl2DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))

class IndexPutImpl2DNoneIndexStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 4], torch.int64, True),
        ([3], torch.int64, True),
        ([1, 3], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (None, index),
                                               value,
                                               accumulate=False,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl2DNoneIndexStaticModule())
def IndexPutImpl2DNoneIndexStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 4, high=3), tu.randint(3, high=3), tu.randint(1, 3, high=1))


class IndexPutImpl3DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=False,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl3DFloatNonAccumulateModule())
def IndexPutImpl3DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPutImpl1DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=False,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl1DIntNonAccumulateModule())
def IndexPutImpl1DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(200, high=1000), tu.randint(300, high=100),
                   tu.randint(300, high=10000))


# ==============================================================================


class IndexPutImpl1DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl1DFloatAccumulateModule())
def IndexPutImpl1DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1000), tu.randint(500, high=10), tu.rand(500))


class IndexPutImpl2DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input.clone(), (index, ),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl2DFloatAccumulateModule())
def IndexPutImpl2DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))


class IndexPutImpl3DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input.clone(), (index, ),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl3DFloatAccumulateModule())
def IndexPutImpl3DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPutImpl1DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(module_factory=lambda: IndexPutImpl1DIntAccumulateModule())
def IndexPutImpl1DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, high=100), tu.randint(10, high=10),
                   tu.randint(10, high=1000))


# ==============================================================================


class IndexPut1DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPut1DFloatNonAccumulateModule())
def IndexPut1DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(100), tu.randint(250, high=100), tu.rand(250))


class IndexPut2DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPut2DFloatNonAccumulateModule())
def IndexPut2DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))


class IndexPut3DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPut3DFloatNonAccumulateModule())
def IndexPut3DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPut1DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(module_factory=lambda: IndexPut1DIntNonAccumulateModule())
def IndexPut1DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(200, high=1000), tu.randint(300, high=100),
                   tu.randint(300, high=10000))


class IndexPut2DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(module_factory=lambda: IndexPut2DIntNonAccumulateModule())
def IndexPut2DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, high=1000))


class IndexPut3DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=False)


@register_test_case(module_factory=lambda: IndexPut3DIntNonAccumulateModule())
def IndexPut3DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, 6, high=1000))


# ==============================================================================


class IndexPut1DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut1DFloatAccumulateModule())
def IndexPut1DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1000), tu.randint(500, high=10), tu.rand(500))


class IndexPut2DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut2DFloatAccumulateModule())
def IndexPut2DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))


class IndexPut3DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut3DFloatAccumulateModule())
def IndexPut3DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPut1DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut1DIntAccumulateModule())
def IndexPut1DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, high=100), tu.randint(10, high=10),
                   tu.randint(10, high=1000))


class IndexPut2DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut2DIntAccumulateModule())
def IndexPut2DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, high=1000))


class IndexPut3DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, (index, ),
                                        value,
                                        accumulate=True)


@register_test_case(module_factory=lambda: IndexPut3DIntAccumulateModule())
def IndexPut3DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, 6, high=1000))


# ==============================================================================
# IndexPutHackedTwin tests are using the aten.index_put.hacked_twin operator.


class IndexPutHackedTwin1DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin1DFloatNonAccumulateModule())
def IndexPutHackedTwin1DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(100), tu.randint(250, high=100), tu.rand(250))


class IndexPutHackedTwin2DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin2DFloatNonAccumulateModule())
def IndexPutHackedTwin2DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))


class IndexPutHackedTwin3DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin3DFloatNonAccumulateModule())
def IndexPutHackedTwin3DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPutHackedTwin1DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin1DIntNonAccumulateModule())
def IndexPutHackedTwin1DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(200, high=1000), tu.randint(300, high=100),
                   tu.randint(300, high=10000))


class IndexPutHackedTwin2DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin2DIntNonAccumulateModule())
def IndexPutHackedTwin2DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, high=1000))


class IndexPutHackedTwin3DIntNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin3DIntNonAccumulateModule())
def IndexPutHackedTwin3DIntNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, 6, high=1000))


# ==============================================================================


class IndexPutHackedTwin1DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin1DFloatAccumulateModule())
def IndexPutHackedTwin1DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1000), tu.randint(500, high=10), tu.rand(500))


class IndexPutHackedTwin2DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin2DFloatAccumulateModule())
def IndexPutHackedTwin2DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8), tu.randint(5, high=4), tu.rand(5, 8))


class IndexPutHackedTwin3DFloatAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin3DFloatAccumulateModule())
def IndexPutHackedTwin3DFloatAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(5, high=4),
                   tu.rand(5, 8, 6))


# ==============================================================================


class IndexPutHackedTwin1DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin1DIntAccumulateModule())
def IndexPutHackedTwin1DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, high=100), tu.randint(10, high=10),
                   tu.randint(10, high=1000))


class IndexPutHackedTwin2DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin2DIntAccumulateModule())
def IndexPutHackedTwin2DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, high=1000))


class IndexPutHackedTwin3DIntAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.index_put(input, [index], value, accumulate=True)


@register_test_case(
    module_factory=lambda: IndexPutHackedTwin3DIntAccumulateModule())
def IndexPutHackedTwin3DIntAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, high=1000), tu.randint(5, high=4),
                   tu.randint(5, 8, 6, high=1000))


# ==============================================================================
# UnsafeIndexPutHackedTwin tests are using the aten._unsafe_index_put.hacked_twin operator.


class UnsafeIndexPutHackedTwin1DFloatNonAccumulateModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._unsafe_index_put(input, [index],
                                        value,
                                        accumulate=False)


@register_test_case(
    module_factory=lambda: UnsafeIndexPutHackedTwin1DFloatNonAccumulateModule())
def UnsafeIndexPutHackedTwin1DFloatNonAccumulateModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(100), tu.randint(250, high=100), tu.rand(250))


# ==============================================================================

class ScatterSrcStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([10, 8, 6], torch.float32, True),
        ([2, 4, 3], torch.int64, True),
        ([5, 8, 6], torch.float32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter(input, 0, index, src)


@register_test_case(
    module_factory=lambda: ScatterSrcStaticModule())
def ScatterSrcStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))

# ==============================================================================

class ScatterSrcModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter(input, 1, index, src)


@register_test_case(
    module_factory=lambda: ScatterSrcModule())
def ScatterSrcModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(3, 4, 3))

# ==============================================================================

class ScatterValueFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
        ([], torch.float64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.scatter(input, 2, index, float(value))


@register_test_case(
    module_factory=lambda: ScatterValueFloatModule())
def ScatterValueFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand().double())

# ==============================================================================

class ScatterValueIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten.scatter(input, 0, index, int(value))


@register_test_case(
    module_factory=lambda: ScatterValueIntModule())
def ScatterValueIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.randint(high=10))

# ==============================================================================

class ScatterReduceFloatModule(torch.nn.Module):
    include_self: bool
    reduce_type: str

    def __init__(self, reduce_type: str, include_self: bool):
        super().__init__()
        self.include_self = include_self
        self.reduce_type = reduce_type

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter_reduce(input, 0, index, src, self.reduce_type, include_self=self.include_self)


@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("sum", False))
def ScatterReduceFloatSumModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("sum", True))
def ScatterReduceFloatSumModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("prod", False))
def ScatterReduceFloatProdModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("prod", True))
def ScatterReduceFloatProdModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amax", False))
def ScatterReduceFloatMaxModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amax", True))
def ScatterReduceFloatMaxModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amin", False))
def ScatterReduceFloatMinModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amin", True))
def ScatterReduceFloatMinModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("mean", False))
def ScatterReduceFloatMeanModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("mean", True))
def ScatterReduceFloatMeanModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))

# ==============================================================================

class ScatterReduceIntModule(torch.nn.Module):
    include_self: bool
    reduce_type: str

    def __init__(self, reduce_type: str, include_self: bool):
        super().__init__()
        self.include_self = include_self
        self.reduce_type = reduce_type

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int32, True),
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.int32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter_reduce(input, 0, index, src, self.reduce_type, include_self=self.include_self)


@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("sum", False))
def ScatterReduceIntSumModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("sum", True))
def ScatterReduceIntSumModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("prod", False))
def ScatterReduceIntProdModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("prod", True))
def ScatterReduceIntProdModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amax", False))
def ScatterReduceIntMaxModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amax", True))
def ScatterReduceIntMaxModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amin", False))
def ScatterReduceIntMinModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amin", True))
def ScatterReduceIntMinModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("mean", False))
def ScatterReduceIntMeanModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("mean", True))
def ScatterReduceIntMeanModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))

# ==============================================================================

class IndexPutImpl2DIndexModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, index, value):
        return torch.ops.aten._index_put_impl_(input, (index, ),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImpl2DIndexModule())
def IndexPutImpl2DIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 7), tu.randint(2, 3, high=3), tu.rand(2, 3, 7))

# ==============================================================================

class IndexPutImplIndexWithNoneModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 3, 4, 5], torch.float32, True),
        ([6, 1], torch.int64, True),
        ([7], torch.int64, True),
        ([2, 3, 6, 7], torch.float32, True),
    ])
    def forward(self, input, index1, index2, value):
        return torch.ops.aten._index_put_impl_(input, (None, None, index1, index2),
                                               value,
                                               accumulate=True,
                                               unsafe=False)


@register_test_case(
    module_factory=lambda: IndexPutImplIndexWithNoneModule())
def IndexPutImplIndexWithNoneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 5), tu.randint(6, 1, high=4), tu.randint(7, high=5), tu.rand(2, 3, 6, 7))

