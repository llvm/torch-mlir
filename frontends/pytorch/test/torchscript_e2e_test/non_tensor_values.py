# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | FileCheck %s

from typing import List, Tuple, Dict

import torch

from torch_mlir_torchscript.e2e_test.framework import run_tests, TestUtils
from torch_mlir_torchscript.e2e_test.reporting import report_results
from torch_mlir_torchscript.e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir_torchscript_e2e_test_configs import TorchScriptTestConfig


class NonTensorValuesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.export
    def test_list(self, x: List[int]) -> List[int]:
        return x

    @torch.jit.export
    def test_tuple(self, x: int) -> Tuple[int, int]:
        return x, x

    @torch.jit.export
    def test_str(self, x: str) -> str:
        return x

    @torch.jit.export
    def test_dict(self, x: Dict[str, int]) -> Dict[str, int]:
        return x


# CHECK: PASS - "NonTensorValuesModule_basic"
@register_test_case(module_factory=lambda: NonTensorValuesModule())
def NonTensorValuesModule_basic(module, tu: TestUtils):
    module.test_list([3])
    module.test_tuple(3)
    module.test_str("hello")
    module.test_dict({"a": 1})


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set())


if __name__ == '__main__':
    main()
