# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir_e2e_test.framework import run_tests, TestUtils
from torch_mlir_e2e_test.reporting import report_results
from torch_mlir_e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.configs import TorchScriptTestConfig


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


# CHECK: Unexpected outcome summary: (myconfig)
# CHECK: FAIL - "MmModule_basic"
# CHECK:     Compilation error:
# Assume that the diagnostic from the TorchScript compiler will at least contain
# the offending "return 3".
# CHECK:     return 3
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(torch.ones([]))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set(), verbose=True, config="myconfig")


if __name__ == "__main__":
    main()
