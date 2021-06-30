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

    def forward(self, t):
        # Static type error that will fail TorchScript compilation -- function
        # that returns tensor along one path and int along another.
        if t.item() > 0:
            return torch.tensor([])
        else:
            return 3


# CHECK: FAIL - "MmModule_basic"
# CHECK:     compilation error
# Assume that the diagnostic from the TorchScript compiler will at least contain
# the offending "return 3".
# CHECK:     return 3
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(torch.ones([]))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set(), verbose=True)


if __name__ == '__main__':
    main()
