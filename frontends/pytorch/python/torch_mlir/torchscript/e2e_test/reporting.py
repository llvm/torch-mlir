#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for reporting the results of the test framework.
"""

from typing import List

import torch

from .framework import TestResult

class _TensorStats:
    def __init__(self, tensor):
        self.min = torch.min(tensor)
        self.max = torch.max(tensor)
        self.mean = torch.mean(tensor)
    def __str__(self):
        return f'min={self.min:+0.4}, max={self.max:+0.4}, mean={self.mean:+0.4f}'


def _print_detailed_tensor_diff(tensor, golden_tensor):
    if tensor.size() != golden_tensor.size():
        print(
            f'Tensor shape mismatch: got {tensor.size()!r}, expected {golden_tensor.size()!r}'
        )
        return
    print('tensor stats       : ', _TensorStats(tensor))
    print('golden tensor stats: ', _TensorStats(golden_tensor))

def report_results(results: List[TestResult]):
    """Provide a basic error report summarizing various TestResult's."""
    any_failed = False
    for result in results:
        failed = False
        for item_num, (item, golden_item) in enumerate(
                zip(result.trace, result.golden_trace)):
            assert item.symbol == golden_item.symbol
            assert len(item.inputs) == len(golden_item.inputs)
            assert len(item.outputs) == len(golden_item.outputs)
            for input, golden_input in zip(item.inputs, golden_item.inputs):
                assert torch.allclose(input, golden_input)
            for output_num, (output, golden_output) in enumerate(
                    zip(item.outputs, golden_item.outputs)):
                # TODO: Refine error message. Things to consider:
                # - Very large tensors -- don't spew, but give useful info
                # - Smaller tensors / primitives -- want to show exact values
                # - Machine parseable format?
                if not torch.allclose(output, golden_output):
                    if not failed:
                        print('FAILURE "{}"'.format(result.unique_name))
                        failed = any_failed = True
                    print(
                        f'Error: in call #{item_num} into the module: result #{output_num} not close in call to "{item.symbol}"'
                    )
                    _print_detailed_tensor_diff(output, golden_output)
        if not failed:
            print('SUCCESS "{}"'.format(result.unique_name))
