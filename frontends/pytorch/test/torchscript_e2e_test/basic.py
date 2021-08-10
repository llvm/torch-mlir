# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir_torchscript.e2e_test.framework import run_tests, TestUtils
from torch_mlir_torchscript.e2e_test.reporting import report_results
from torch_mlir_torchscript.e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir_torchscript_e2e_test_configs import TorchScriptTestConfig


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


# TODO: Refine messages.
# CHECK: PASS - "MmModule_basic"
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


# CHECK: PASS - "MmModule_basic2"
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic2(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set(), verbose=True)


if __name__ == '__main__':
    main()
