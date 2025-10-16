# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir_e2e_test.framework import run_tests, TestUtils
from torch_mlir_e2e_test.reporting import report_results
from torch_mlir_e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.configs.torchscript import TorchScriptTestConfig


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        # Input of `torch.tensor` only allows ints, floats, or bools while empty
        # list defaults to tensor type
        return torch.tensor([])


# CHECK: FAIL - "MmModule_basic"
# CHECK:     Runtime error:
# Assume that the diagnostic from the TorchScript runtime will at least contain
# the offending "return torch.tensor([])".
# CHECK:     return torch.tensor([])
@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(torch.ones([]))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set(), verbose=True)


if __name__ == "__main__":
    main()
