# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Utilities for reporting the results of the test framework.
"""

from typing import Any, List, Optional, Set

import collections
import io
import textwrap

import torch

from .framework import TestResult, TraceItem


class TensorSummary:
    """A summary of a tensor's contents."""
    def __init__(self, tensor):
        self.min = torch.min(tensor.type(torch.float64))
        self.max = torch.max(tensor.type(torch.float64))
        self.mean = torch.mean(tensor.type(torch.float64))
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __str__(self):
        return f'Tensor with shape={self.shape}, dtype={self.dtype}, min={self.min:+0.4}, max={self.max:+0.4}, mean={self.mean:+0.4}'


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
    """
    def __init__(self, value, golden_value, context: ErrorContext):
        self.value = value
        self.golden_value = golden_value
        self.context = context
        self.failure_reasons = []
        self._evaluate_outcome()

    @property
    def failed(self):
        return len(self.failure_reasons) != 0

    def error_str(self):
        return '\n'.join(self.failure_reasons)

    def _evaluate_outcome(self):
        value, golden = self.value, self.golden_value
        if isinstance(golden, float):
            if not isinstance(value, float):
                return self._record_mismatch_type_failure('float', value)
            if abs(value - golden) / golden > 1e-4:
                return self._record_failure(
                    f'value ({value!r}) is not close to golden value ({golden!r})'
                )
            return
        if isinstance(golden, int):
            if not isinstance(value, int):
                return self._record_mismatch_type_failure('int', value)
            if value != golden:
                return self._record_failure(
                    f'value ({value!r}) is not equal to golden value ({golden!r})'
                )
            return
        if isinstance(golden, str):
            if not isinstance(value, str):
                return self._record_mismatch_type_failure('str', value)
            if value != golden:
                return self._record_failure(
                    f'value ({value!r}) is not equal to golden value ({golden!r})'
                )
            return
        if isinstance(golden, tuple):
            if not isinstance(value, tuple):
                return self._record_mismatch_type_failure('tuple', value)
            if len(value) != len(golden):
                return self._record_failure(
                    f'value ({len(value)!r}) is not equal to golden value ({len(golden)!r})'
                )
            reports = [
                ValueReport(v, g, self.context.chain(f'tuple element {i}'))
                for i, (v, g) in enumerate(zip(value, golden))
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, list):
            if not isinstance(value, list):
                return self._record_mismatch_type_failure('list', value)
            if len(value) != len(golden):
                return self._record_failure(
                    f'value ({len(value)!r}) is not equal to golden value ({len(golden)!r})'
                )
            reports = [
                ValueReport(v, g, self.context.chain(f'list element {i}'))
                for i, (v, g) in enumerate(zip(value, golden))
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, dict):
            if not isinstance(value, dict):
                return self._record_mismatch_type_failure('dict', value)
            gkeys = list(sorted(golden.keys()))
            vkeys = list(sorted(value.keys()))
            if gkeys != vkeys:
                return self._record_failure(
                    f'dict keys ({vkeys!r}) are not equal to golden keys ({gkeys!r})'
                )
            reports = [
                ValueReport(value[k], golden[k],
                            self.context.chain(f'dict element at key {k!r}'))
                for k in gkeys
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                return self._record_mismatch_type_failure('torch.Tensor', value)

            if value.shape != golden.shape:
                return self._record_failure(
                    f'shape ({value.shape}) is not equal to golden shape ({golden.shape})'
                )
            if value.dtype != golden.dtype:
                return self._record_failure(
                    f'dtype ({value.dtype}) is not equal to golden dtype ({golden.dtype})'
                )
            if not torch.allclose(value, golden, rtol=1e-03, atol=1e-07, equal_nan=True):
                return self._record_failure(
                    f'value ({TensorSummary(value)}) is not close to golden value ({TensorSummary(golden)})'
                )
            return
        return self._record_failure(
            f'unexpected golden value of type `{golden.__class__.__name__}`')

    def _record_failure(self, s: str):
        self.failure_reasons.append(self.context.format_error(s))

    def _record_mismatch_type_failure(self, expected: str, actual: Any):
        self._record_failure(
            f'expected a value of type `{expected}` but got `{actual.__class__.__name__}`'
        )



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
        for i, (input, golden_input) in enumerate(
                zip(self.item.inputs, self.golden_item.inputs)):
            value_report = ValueReport(
                input, golden_input,
                self.context.chain(
                    f'input #{i} of call to "{self.item.symbol}"'))
            if value_report.failed:
                self.failure_reasons.append(value_report.error_str())
        value_report = ValueReport(
            self.item.output, self.golden_item.output,
            self.context.chain(f'output of call to "{self.item.symbol}"'))
        if value_report.failed:
            self.failure_reasons.append(value_report.error_str())


class SingleTestReport:
    """A report for a single test."""
    item_reports: Optional[List[TraceItemReport]]

    def __init__(self, result: TestResult, context: ErrorContext):
        self.result = result
        self.context = context
        self.item_reports = None
        if result.compilation_error is None and result.runtime_error is None:
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
        elif self.result.runtime_error is not None:
            return True
        return any(r.failed for r in self.item_reports)

    def error_str(self):
        assert self.failed
        f = io.StringIO()
        p = lambda *x: print(*x, file=f)
        if self.result.compilation_error is not None:
            return 'Compilation error: ' + self.result.compilation_error
        elif self.result.runtime_error is not None:
            return 'Runtime error: ' + self.result.runtime_error
        for report in self.item_reports:
            if report.failed:
                p(report.error_str())
        return f.getvalue()


def report_results(results: List[TestResult],
                   expected_failures: Set[str],
                   verbose: bool = False):
    """Print a basic error report summarizing various TestResult's.

    This report uses the PASS/FAIL/XPASS/XFAIL nomenclature of LLVM's
    "lit" testing utility. See
    https://llvm.org/docs/CommandGuide/lit.html#test-status-results

    The `expected_failures` set should contain the names of tests
    (according to their `unique_name`) which are expected to fail.
    The overall passing/failing status of the report requires these to fail
    in order to succeed (this catches cases where things suddenly
    start working).

    If `verbose` is True, then provide an explanation of what failed.

    Returns True if the run resulted in any unexpected pass/fail behavior.
    Otherwise False.
    """
    results_by_outcome = collections.defaultdict(list)
    for result in results:
        report = SingleTestReport(result, ErrorContext.empty())
        expected_failure = result.unique_name in expected_failures
        if expected_failure:
            if report.failed:
                print(f'XFAIL - "{result.unique_name}"')
                results_by_outcome['XFAIL'].append((result, report))
            else:
                print(f'XPASS - "{result.unique_name}"')
                results_by_outcome['XPASS'].append((result, report))
        else:
            if not report.failed:
                print(f'PASS - "{result.unique_name}"')
                results_by_outcome['PASS'].append((result, report))
            else:
                print(f'FAIL - "{result.unique_name}"')
                results_by_outcome['FAIL'].append((result, report))

    OUTCOME_MEANINGS = collections.OrderedDict()
    OUTCOME_MEANINGS['PASS'] = 'Passed'
    OUTCOME_MEANINGS['FAIL'] = 'Failed'
    OUTCOME_MEANINGS['XFAIL'] = 'Expectedly Failed'
    OUTCOME_MEANINGS['XPASS'] = 'Unexpectedly Passed'

    had_unexpected_results = len(results_by_outcome['FAIL']) != 0 or len(
        results_by_outcome['XPASS']) != 0

    if had_unexpected_results:
        print('\nUnexpected outcome summary:')

    # For FAIL and XPASS (unexpected outcomes), print a summary.
    for outcome, results in results_by_outcome.items():
        # PASS and XFAIL are "good"/"successful" outcomes.
        if outcome == 'PASS' or outcome == 'XFAIL':
            continue
        # If there is nothing to report, be quiet.
        if len(results) == 0:
            continue
        print(f'\n****** {OUTCOME_MEANINGS[outcome]} tests - {len(results)} tests')
        for result, report in results:
            print(f'    {outcome} - "{result.unique_name}"')
            # If the test failed, print the error message.
            if outcome == 'FAIL' and verbose:
                print(textwrap.indent(report.error_str(), ' ' * 8))

    # Print a summary for easy scanning.
    print('\nSummary:')

    for key in ['PASS', 'FAIL', 'XFAIL', 'XPASS']:
        if results_by_outcome[key]:
            print(f'    {OUTCOME_MEANINGS[key]}: {len(results_by_outcome[key])}')
    return had_unexpected_results
