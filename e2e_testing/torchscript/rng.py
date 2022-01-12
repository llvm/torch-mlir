import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export


# ==============================================================================
class UniformModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float64, True),
        ([-1, -1], torch.float64, True),
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, x, y, z):
        a = torch.ops.aten.uniform_(x, 1.0, 10.0)
        b = torch.ops.aten.uniform_(y, -20.0, -5.0)
        c = torch.ops.aten.uniform_(z, -15.0, 3.0)
        std = torch.cat([
            torch.flatten(torch.std(a)),
            torch.flatten(torch.std(b)),
            torch.flatten(torch.std(c))
        ])
        mean = torch.cat([
            torch.flatten(torch.mean(a)),
            torch.flatten(torch.mean(b)),
            torch.flatten(torch.mean(c))
        ])
        return std, mean


@register_test_case(module_factory=lambda: UniformModule())
def UniformModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(256, 512, 64).double(),
        tu.rand(512, 1024, 128).double(),
        tu.rand(512, 256, 1024).double())

