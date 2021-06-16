#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Utilities for reporting the results of the test framework.
"""

from typing import List, Optional

import io
import textwrap

import torch

from .framework import TestResult, TraceItem


class TensorSummary:
    """A summary of a tensor's contents."""
    def __init__(self, tensor):
        self.min = torch.min(tensor)
        self.max = torch.max(tensor)
        self.mean = torch.mean(tensor)

    def __str__(self):
        return f'Tensor with min={self.min:+0.4}, max={self.max:+0.4}, mean={self.mean:+0.4f}'


class ErrorContext:
    """A chained list of error contexts.

    This is useful for tracking errors across multiple levels of detail.
    """
    def __init__(self, contexts: List[str]):
        self.contexts = contexts

    @staticmethod
    def empty():
        """Create an empty error context.

        Used as the top-level context.
        """
        return ErrorContext([])

    def chain(self, additional_context: str):
        """Chain an additional context onto the current error context.
        """
        return ErrorContext(self.contexts + [additional_context])

    def format_error(self, s: str):
        return '@ ' + '\n@ '.join(self.contexts) + '\n' + 'ERROR: ' + s


class ValueReport:
    """A report for a single value processed by the program.

    This is currently limited to tensors, but eventually will support
    all legal TorchScript types.
    """
    def __init__(self, value, golden_value, context: ErrorContext):
        self.value = value
        self.golden_value = golden_value
        self.context = context

    @property
    def failed(self):
        return not torch.allclose(self.value, self.golden_value, rtol=1e-04, atol=1e-08)

    def error_str(self):
        assert self.failed
        if self.value.size() != self.golden_value.size():
            return self.context.format_error(
                f'tensor shape mismatch: got {tensor.size()!r}, expected {golden_tensor.size()!r}'
            )
        f = io.StringIO()
        p = lambda *x: print(*x, file=f)
        p('values mismatch')
        p('got     : ', TensorSummary(self.value))
        p('expected: ', TensorSummary(self.golden_value))
        return self.context.format_error(f.getvalue())


class TraceItemReport:
    """A report for a single trace item."""
    failure_reasons: List[str]

    def __init__(self, item: TraceItem, golden_item: TraceItem,
                 context: ErrorContext):
        self.item = item
        self.golden_item = golden_item
        self.context = context
        self.failure_reasons = []
        self._evaluate_outcome()

    @property
    def failed(self):
        return len(self.failure_reasons) != 0

    def error_str(self):
        return '\n'.join(self.failure_reasons)

    def _evaluate_outcome(self):
        if self.item.symbol != self.golden_item.symbol:
            self.failure_reasons.append(
                self.context.format_error(
                    f'not invoking the same symbol: got "{self.item.symbol}", expected "{self.golden_item.symbol}"'
                ))
        if len(self.item.inputs) != len(self.golden_item.inputs):
            self.failure_reasons.append(
                self.context.format_error(
                    f'different number of inputs: got "{len(self.item.inputs)}", expected "{len(self.golden_item.inputs)}"'
                ))
        if len(self.item.outputs) != len(self.golden_item.outputs):
            self.failure_reasons.append(
                self.context.format_error(
                    f'different number of outputs: got "{len(self.item.outputs)}", expected "{len(self.golden_item.outputs)}"'
                ))
        for i, (input, golden_input) in enumerate(
                zip(self.item.inputs, self.golden_item.inputs)):
            value_report = ValueReport(
                input, golden_input,
                self.context.chain(
                    f'input #{i} of call to "{self.item.symbol}"'))
            if value_report.failed:
                self.failure_reasons.append(value_report.error_str())
        for i, (output, golden_output) in enumerate(
                zip(self.item.outputs, self.golden_item.outputs)):
            value_report = ValueReport(output, golden_output,
                                       self.context.chain(f'output #{i}'))
            if value_report.failed:
                self.failure_reasons.append(value_report.error_str())


class SingleTestReport:
    """A report for a single test."""
    item_reports: Optional[List[TraceItemReport]]

    def __init__(self, result: TestResult, context: ErrorContext):
        self.result = result
        self.context = context
        self.item_reports = None
        if result.compilation_error is None:
            self.item_reports = []
            for i, (item, golden_item) in enumerate(
                    zip(result.trace, result.golden_trace)):
                self.item_reports.append(
                    TraceItemReport(
                        item, golden_item,
                        context.chain(
                            f'trace item #{i} - call to "{item.symbol}"')))

    @property
    def failed(self):
        if self.result.compilation_error is not None:
            return True
        return any(r.failed for r in self.item_reports)

    def error_str(self):
        assert self.failed
        f = io.StringIO()
        p = lambda *x: print(*x, file=f)
        if self.result.compilation_error is not None:
            return 'compilation error' + self.result.compilation_error
        for report in self.item_reports:
            if report.failed:
                p(report.error_str())
        return f.getvalue()


def report_results(results: List[TestResult], verbose: bool = False):
    """Provide a basic error report summarizing various TestResult's.

    If `verbose` is True, then provide an explanation of what failed.
    """
    for result in results:
        report = SingleTestReport(result, ErrorContext.empty())
        if not report.failed:
            print(f'SUCCESS - "{result.unique_name}"')
        else:
            error_str = ''
            if verbose:
                error_str = '\n' + textwrap.indent(report.error_str(), '    ')
            print(f'FAILURE - "{result.unique_name}"' + error_str)
