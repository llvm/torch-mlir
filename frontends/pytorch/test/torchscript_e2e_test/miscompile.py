# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir.torchscript.e2e_test.framework import run_tests, TestUtils
from torch_mlir.torchscript.e2e_test.reporting import report_results
from torch_mlir.torchscript.e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir.torchscript.e2e_test.configs import TorchScriptTestConfig


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        # Use torch.jit.is_scripting() to fake a miscompile.
        # The non-scripted code will take one path, and the scripted code
        # will take another path.
        if torch.jit.is_scripting():
            return torch.mm(rhs, lhs)
        return torch.mm(lhs, rhs)


# TODO: Refine error messages.
# CHECK: FAILURE "MmModule_basic"
# CHECK: Error: in call #0 into the module: result #0 not close in call to "forward"
# CHECK: tensor stats       :  min={{.*}}, max={{.*}}, mean={{.*}}
# CHECK: golden tensor stats:  min={{.*}}, max={{.*}}, mean={{.*}}
# CHECK-NOT: ALL PASS
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results)


if __name__ == '__main__':
    main()
