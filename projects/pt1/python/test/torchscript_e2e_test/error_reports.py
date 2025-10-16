# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import List, Tuple, Dict

import torch

from torch_mlir_e2e_test.framework import run_tests, TestUtils
from torch_mlir_e2e_test.reporting import report_results
from torch_mlir_e2e_test.registry import register_test_case, GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.configs.torchscript import TorchScriptTestConfig

# CHECK: Unexpected outcome summary:
# CHECK: FAIL - "ErroneousModule_basic"


# Use torch.jit.is_scripting() to fake miscompiles.
# The non-scripted code will take one path, and the scripted code
# will take another path.
class ErroneousModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # CHECK-NEXT: @ trace item #0 - call to "test_int"
    # CHECK-NEXT: @ output of call to "test_int"
    # CHECK-NEXT: ERROR: value (1) is not equal to golden value (2)
    @torch.jit.export
    def test_int(self) -> int:
        if torch.jit.is_scripting():
            return 1
        else:
            return 2

    # CHECK-NEXT: @ trace item #1 - call to "test_float"
    # CHECK-NEXT: @ output of call to "test_float"
    # CHECK-NEXT: ERROR: value (1.0) is not close to golden value (2.0)
    @torch.jit.export
    def test_float(self) -> float:
        if torch.jit.is_scripting():
            return 1.0
        else:
            return 2.0

    # CHECK-NEXT: @ trace item #2 - call to "test_list_element"
    # CHECK-NEXT: @ output of call to "test_list_element"
    # CHECK-NEXT: @ list element 2
    # CHECK-NEXT: ERROR: value (3) is not equal to golden value (4)
    @torch.jit.export
    def test_list_element(self) -> List[int]:
        if torch.jit.is_scripting():
            return [1, 2, 3]
        else:
            return [1, 2, 4]

    # CHECK-NEXT: @ trace item #3 - call to "test_tuple_element"
    # CHECK-NEXT: @ output of call to "test_tuple_element"
    # CHECK-NEXT: @ tuple element 0
    # CHECK-NEXT: ERROR: value (1) is not equal to golden value (2)
    # CHECK-NEXT: @ trace item #3 - call to "test_tuple_element"
    # CHECK-NEXT: @ output of call to "test_tuple_element"
    # CHECK-NEXT: @ tuple element 1
    # CHECK-NEXT: ERROR: value (2) is not equal to golden value (3)
    @torch.jit.export
    def test_tuple_element(self) -> Tuple[int, int]:
        if torch.jit.is_scripting():
            return (1, 2)
        else:
            return (2, 3)

    # CHECK-NEXT: @ trace item #4 - call to "test_str"
    # CHECK-NEXT: @ output of call to "test_str"
    # CHECK-NEXT: ERROR: value ('x') is not equal to golden value ('y')
    @torch.jit.export
    def test_str(self) -> str:
        if torch.jit.is_scripting():
            return "x"
        else:
            return "y"

    # CHECK-NEXT: @ trace item #5 - call to "test_dict_keys"
    # CHECK-NEXT: @ output of call to "test_dict_keys"
    # CHECK-NEXT: ERROR: dict keys (['x']) are not equal to golden keys (['y'])
    @torch.jit.export
    def test_dict_keys(self) -> Dict[str, int]:
        if torch.jit.is_scripting():
            return {"x": 1}
        else:
            return {"y": 21}

    # CHECK-NEXT: @ trace item #6 - call to "test_dict_values"
    # CHECK-NEXT: @ output of call to "test_dict_values"
    # CHECK-NEXT: @ dict element at key 'x'
    # CHECK-NEXT: ERROR: value (1) is not equal to golden value (2)
    @torch.jit.export
    def test_dict_values(self) -> Dict[str, int]:
        if torch.jit.is_scripting():
            return {"x": 1}
        else:
            return {"x": 2}

    # CHECK-NEXT: @ trace item #7 - call to "test_recursive"
    # CHECK-NEXT: @ output of call to "test_recursive"
    # CHECK-NEXT: @ list element 0
    # CHECK-NEXT: @ dict element at key 'x'
    # CHECK-NEXT: @ list element 0
    # CHECK-NEXT: @ list element 1
    # CHECK-NEXT: ERROR: value (3) is not equal to golden value (4)
    @torch.jit.export
    def test_recursive(self):
        if torch.jit.is_scripting():
            return [({"x": [[2, 3]]})]
        else:
            return [({"x": [[2, 4]]})]

    # CHECK-NEXT: @ trace item #8 - call to "test_tensor_value_mismatch"
    # CHECK-NEXT: @ output of call to "test_tensor_value_mismatch"
    # CHECK-NEXT: ERROR: value (Tensor with shape=[3], dtype=torch.float32, min=+1.0, max=+3.0, mean=+2.0) is not close to golden value (Tensor with shape=[3], dtype=torch.float32, min=+1.5, max=+3.5, mean=+2.5)
    @torch.jit.export
    def test_tensor_value_mismatch(self):
        if torch.jit.is_scripting():
            return torch.tensor([1.0, 2.0, 3.0])
        else:
            return torch.tensor([1.5, 2.5, 3.5])

    # CHECK-NEXT: @ trace item #9 - call to "test_tensor_shape_mismatch"
    # CHECK-NEXT: @ output of call to "test_tensor_shape_mismatch"
    # CHECK-NEXT: ERROR: shape (torch.Size([2])) is not equal to golden shape (torch.Size([3]))
    @torch.jit.export
    def test_tensor_shape_mismatch(self):
        if torch.jit.is_scripting():
            return torch.tensor([1.0, 2.0])
        else:
            return torch.tensor([1.0, 2.0, 3.0])


@register_test_case(module_factory=lambda: ErroneousModule())
def ErroneousModule_basic(module, tu: TestUtils):
    module.test_int()
    module.test_float()
    module.test_list_element()
    module.test_tuple_element()
    module.test_str()
    module.test_dict_keys()
    module.test_dict_values()
    module.test_recursive()
    module.test_tensor_value_mismatch()
    module.test_tensor_shape_mismatch()


def main():
    config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results, set(), verbose=True)


if __name__ == "__main__":
    main()
