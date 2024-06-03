import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class RandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([1024, 512], torch.float, True)])
    def forward(self, x):
        size = x.size()
        a = torch.rand(size)
        return torch.std(a), torch.mean(a)


@register_test_case(module_factory=lambda: RandModule())
def RandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 512))


# ==============================================================================


class UniformModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x, y, z):
        a = torch.ops.aten.uniform_(x, 1.0, 10.0)
        b = torch.ops.aten.uniform_(y, -20.0, -5.0)
        c = torch.ops.aten.uniform_(z, -15.0, 3.0)
        std = torch.cat(
            [
                torch.flatten(torch.std(a)),
                torch.flatten(torch.std(b)),
                torch.flatten(torch.std(c)),
            ]
        )
        mean = torch.cat(
            [
                torch.flatten(torch.mean(a)),
                torch.flatten(torch.mean(b)),
                torch.flatten(torch.mean(c)),
            ]
        )
        return std, mean


@register_test_case(module_factory=lambda: UniformModule())
def UniformModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(256, 512, 12).double(),
        tu.rand(512, 1024, 12).double(),
        tu.rand(512, 256, 12).double(),
    )


# ==============================================================================


class UniformStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([256, 512, 12], torch.float64, True),
            ([512, 1024, 12], torch.float64, True),
            ([512, 256, 12], torch.float64, True),
        ]
    )
    def forward(self, x, y, z):
        a = torch.ops.aten.uniform_(x, 1.0, 10.0)
        b = torch.ops.aten.uniform_(y, -20.0, -5.0)
        c = torch.ops.aten.uniform_(z, -15.0, 3.0)
        std = torch.cat(
            [
                torch.flatten(torch.std(a)),
                torch.flatten(torch.std(b)),
                torch.flatten(torch.std(c)),
            ]
        )
        mean = torch.cat(
            [
                torch.flatten(torch.mean(a)),
                torch.flatten(torch.mean(b)),
                torch.flatten(torch.mean(c)),
            ]
        )
        return std, mean


@register_test_case(module_factory=lambda: UniformStaticShapeModule())
def UniformStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(256, 512, 12).double(),
        tu.rand(512, 1024, 12).double(),
        tu.rand(512, 256, 12).double(),
    )


# ==============================================================================


class UniformNoCorrelationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def correlation(self, stack):
        """Calculate the correlation of the two rows in `stack`.

        TODO: Remove this by adding support for `torch.corrcoef`.
        """
        m = stack - torch.mean(stack, dim=1, keepdim=True)
        cov = torch.matmul(m, m.t()) / (stack.size()[1] - 1)
        return cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1])

    @export
    @annotate_args(
        [
            None,
            ([1000], torch.float64, True),
        ]
    )
    def forward(self, x):
        # Correlation of two independent uniforms
        a = torch.ops.aten.uniform(x)
        b = torch.ops.aten.uniform(x)
        stack = torch.cat((a.unsqueeze(0), b.unsqueeze(0)))
        corr_a_b = self.correlation(stack)

        # Split first dimension of large buffer
        larger = torch.empty((2,) + x.size(), dtype=torch.float64)
        larger = torch.ops.aten.uniform(larger)
        corr_major = self.correlation(larger)

        # Split second dimension of large buffer
        corr_minor = self.correlation(larger.t().reshape(2, -1))

        # This is hacky. The problem with returning just the correlations
        # is that `torch.allclose` becomes stricter the closer values are to
        # zero. Since the correlations are on the order of 1E-3, `rtol=1E-3`,
        # and `atol=1E-7`, the correlations have to have a difference smaller
        # than `atol + rtol * correlation = 1E-6`, which is too strict.
        # Instead, the correlations are explicitly required to be less than
        # 0.001.
        return (
            torch.where(torch.abs(corr_a_b) < 0.001, 1, 2),
            torch.where(torch.abs(corr_major) < 0.001, 1, 2),
            torch.where(torch.abs(corr_minor) < 0.001, 1, 2),
        )


@register_test_case(module_factory=lambda: UniformNoCorrelationModule())
def UniformNoCorrelationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1000).double())


# ==============================================================================


class ExponentialModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.exponential(x, 3.0)
        mean = torch.mean(a)
        std = torch.std(a)
        return mean, std


@register_test_case(module_factory=lambda: ExponentialModule())
def ExponentialModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 512, 16).double())


# ==============================================================================


class BernoulliModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.bernoulli(x)
        mean = torch.mean(a)
        std = torch.std(a)
        return mean, std


@register_test_case(module_factory=lambda: BernoulliModule())
def BernoulliModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 512, 16).double())


# ==============================================================================


class BernoulliZerosModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        return torch.bernoulli(x)


@register_test_case(module_factory=lambda: BernoulliZerosModule())
def BernoulliZerosModule_basic(module, tu: TestUtils):
    module.forward(torch.zeros(4, 8).double())


# ==============================================================================


class BernoulliOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        return torch.bernoulli(x)


@register_test_case(module_factory=lambda: BernoulliOnesModule())
def BernoulliOnesModule_basic(module, tu: TestUtils):
    module.forward(torch.ones(4, 8).double())


# ==============================================================================


class BernoulliFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x, y):
        a = torch.ops.aten.bernoulli_(x, 0.4)
        b = torch.ops.aten.bernoulli_(y, 0.7)
        mean = torch.cat(
            [
                torch.flatten(torch.mean(a)),
                torch.flatten(torch.mean(b)),
            ]
        )
        std = torch.cat(
            [
                torch.flatten(torch.std(a)),
                torch.flatten(torch.std(b)),
            ]
        )
        return mean, std


@register_test_case(module_factory=lambda: BernoulliFloatModule())
def BernoulliFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 512, 16).double(), tu.rand(512, 512, 16).double())


# ==============================================================================


class BernoulliTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x, px):
        a = torch.ops.aten.bernoulli_(x, px)
        mean = torch.mean(a)
        std = torch.std(a)
        return mean, std


@register_test_case(module_factory=lambda: BernoulliTensorModule())
def BernoulliTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 512).double(), tu.rand(512, 512).double())


# ==============================================================================


class BernoulliPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x, y):
        a = torch.ops.aten.bernoulli(x, 0.4)
        b = torch.ops.aten.bernoulli(y, 0.7)
        mean = torch.cat(
            [
                torch.flatten(torch.mean(a)),
                torch.flatten(torch.mean(b)),
            ]
        )
        std = torch.cat(
            [
                torch.flatten(torch.std(a)),
                torch.flatten(torch.std(b)),
            ]
        )
        return mean, std


@register_test_case(module_factory=lambda: BernoulliPModule())
def BernoulliPModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 512, 16).double(), tu.rand(512, 512, 16).double())


# ==============================================================================


class MultinomialModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.multinomial(x, 1024*1024, replacement=True)
        return a.mean(dtype=torch.double)


@register_test_case(module_factory=lambda: MultinomialModule())
def MultinomialModule_basic(module, tu: TestUtils):
    x = tu.rand(100).double()
    module.forward(x)

class MultinomialModule2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.multinomial(x, 1024*1024, replacement=True)
        return a.mean(dtype=torch.double)


@register_test_case(module_factory=lambda: MultinomialModule2D())
def MultinomialModule2D_basic(module, tu: TestUtils):
    x = tu.rand(10, 100).double()
    module.forward(x)


# ==============================================================================

class RandLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.rand_like(x)
        mean = torch.mean(a)
        return mean


@register_test_case(module_factory=lambda: RandLikeModule())
def RandLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1024).double())


# ==============================================================================


class RandLikeDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.rand_like(x, dtype=torch.float32)
        mean = torch.mean(a)
        return mean


@register_test_case(module_factory=lambda: RandLikeDtypeModule())
def RandLikeDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1024).double())


# ==============================================================================


class RandIntLowModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randint(low=1, high=1000, size=[1024, 1024])
        mean = torch.mean(a.to(torch.float32))
        return mean


@register_test_case(module_factory=lambda: RandIntLowModule())
def RandIntLowModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandIntLowDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randint(
            low=1, high=1000, size=[128, 256, 512], dtype=torch.float64
        )
        mean = torch.mean(a)
        return mean


@register_test_case(module_factory=lambda: RandIntLowDtypeModule())
def RandIntLowDtypeModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randint(high=1000, size=[1024, 1024])
        mean = torch.mean(a.to(torch.float32))
        return mean


@register_test_case(module_factory=lambda: RandIntModule())
def RandIntModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandIntDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randint(high=1000, size=[128, 256, 512], dtype=torch.float64)
        mean = torch.mean(a.to(torch.float32))
        return mean


@register_test_case(module_factory=lambda: RandIntDtypeModule())
def RandIntDtypeModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandIntPinMemoryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randint(high=1000, size=[128, 256, 512], pin_memory=False)
        mean = torch.mean(a.to(torch.float32))
        return mean


@register_test_case(module_factory=lambda: RandIntPinMemoryModule())
def RandIntPinMemoryModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randn([4, 512, 1024])
        std = torch.std(a.to(dtype=torch.float64))
        return std


@register_test_case(module_factory=lambda: RandnModule())
def RandnModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandnDtypeDeviceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randn(
            [4, 512, 1024], dtype=torch.float64, device=torch.device("cpu")
        )
        std = torch.std(a)
        return std


@register_test_case(module_factory=lambda: RandnDtypeDeviceModule())
def RandnDtypeDeviceModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandnGeneratorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randn([4, 512, 1024], generator=None)
        std = torch.std(a.to(dtype=torch.float64))
        return std


@register_test_case(module_factory=lambda: RandnGeneratorModule())
def RandnGeneratorModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandnGeneratorF64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = torch.ops.aten.randn([4, 512, 1024], generator=None, dtype=torch.float64)
        std = torch.std(a)
        return std


@register_test_case(module_factory=lambda: RandnGeneratorF64Module())
def RandnGeneratorF64Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class RandnLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.randn_like(x)
        std = torch.std(a)
        return std


@register_test_case(module_factory=lambda: RandnLikeModule())
def RandnLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 512, 1024).double())


# ==============================================================================


class RandnLikeDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.randn_like(x, dtype=torch.float32)
        std = torch.std(a)
        return std


@register_test_case(module_factory=lambda: RandnLikeDtypeModule())
def RandnLikeDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256, 1024).double())


# ==============================================================================


class NormalFunctionalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.normal_functional(x, mean=-5.0, std=2.0)
        mean = torch.mean(a)
        std = torch.std(a)
        return mean, std


@register_test_case(module_factory=lambda: NormalFunctionalModule())
def NormalFunctionalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2048, 4096).double())
