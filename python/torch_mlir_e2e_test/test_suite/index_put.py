# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

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
    module.forward(tu.rand(100), torch.randint(100, (250, )), tu.rand(250))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(1000, (200, )), torch.randint(100, (300, )),
                   torch.randint(10000, (300, )))


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
    module.forward(tu.rand(1000), torch.randint(10, (500, )), tu.rand(500))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(100, (10, )), torch.randint(10, (10, )),
                   torch.randint(1000, (10, )))


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
    module.forward(tu.rand(100), torch.randint(100, (250, )), tu.rand(250))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(1000, (200, )), torch.randint(100, (300, )),
                   torch.randint(10000, (300, )))


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
    module.forward(torch.randint(1000, (10, 8)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8)))


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
    module.forward(torch.randint(1000, (10, 8, 6)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8, 6)))


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
    module.forward(tu.rand(1000), torch.randint(10, (500, )), tu.rand(500))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(100, (10, )), torch.randint(10, (10, )),
                   torch.randint(1000, (10, )))


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
    module.forward(torch.randint(1000, (10, 8)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8)))


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
    module.forward(torch.randint(1000, (10, 8, 6)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8, 6)))


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
    module.forward(tu.rand(100), torch.randint(100, (250, )), tu.rand(250))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(1000, (200, )), torch.randint(100, (300, )),
                   torch.randint(10000, (300, )))


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
    module.forward(torch.randint(1000, (10, 8)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8)))


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
    module.forward(torch.randint(1000, (10, 8, 6)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8, 6)))


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
    module.forward(tu.rand(1000), torch.randint(10, (500, )), tu.rand(500))


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
    module.forward(tu.rand(10, 8), torch.randint(4, (5, )), tu.rand(5, 8))


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
    module.forward(tu.rand(10, 8, 6), torch.randint(4, (5, )),
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
    module.forward(torch.randint(100, (10, )), torch.randint(10, (10, )),
                   torch.randint(1000, (10, )))


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
    module.forward(torch.randint(1000, (10, 8)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8)))


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
    module.forward(torch.randint(1000, (10, 8, 6)), torch.randint(4, (5, )),
                   torch.randint(1000, (5, 8, 6)))
