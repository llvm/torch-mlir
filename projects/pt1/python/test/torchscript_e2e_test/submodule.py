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


class Submodule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m2 = Submodule2()


class ModuleWithSubmodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Submodule()


# CHECK: PASS - "ModuleWithSubmodule_basic"
@register_test_case(module_factory=lambda: ModuleWithSubmodule())
def ModuleWithSubmodule_basic(module, tu: TestUtils):
    module.m.m2.forward(tu.rand(4, 4), tu.rand(4, 4))


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set())


if __name__ == "__main__":
    main()
