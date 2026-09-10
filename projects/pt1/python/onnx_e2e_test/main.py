# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Entry point for the ONNX-native end-to-end test suite.

Runs every ``OnnxTest`` in ``GLOBAL_ONNX_TEST_REGISTRY`` through the
selected config, compares outputs against the ONNX reference evaluator, and
reports results using the existing ``torch_mlir_e2e_test.reporting`` layer.

Usage::

    python -m onnx_e2e_test.main --config linalg -v --filter Add
"""

import argparse
import re
import sys
import traceback
from typing import List

import torch

from torch_mlir_e2e_test.framework import TestResult, TraceItem
from torch_mlir_e2e_test.reporting import report_results

from .config import (
    OnnxLinalgTestConfig,
    OnnxTosaBackendTestConfig,
    materialize_inputs,
)
from .registry import GLOBAL_ONNX_TEST_REGISTRY
from .xfail_sets import (
    ONNX_E2E_LINALG_CRASHING_SET,
    ONNX_E2E_TOSA_CRASHING_SET,
    ONNX_E2E_TOSA_XFAIL_SET,
    ONNX_E2E_LINALG_XFAIL_SET,
)

# Register built-in test cases (side-effecting import).
from .test_suite import register_all_tests as _register_all_tests

_register_all_tests()

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

_SYMBOL = "main_graph"


def _run_one(test, config: OnnxLinalgTestConfig) -> TestResult:
    """Run a single OnnxTest and return a TestResult.

    The comparison is delegated to report_results (via TraceItem / golden_trace),
    which uses torch.allclose with rtol=1e-03, atol=1e-07 like the rest of the
    e2e suite.
    """
    # Build the model (may raise if the factory is broken).
    try:
        model_proto = test.model_factory()
    except Exception:
        return TestResult(
            unique_name=test.unique_name,
            compilation_error=traceback.format_exc(),
            runtime_error=None,
            trace=None,
            golden_trace=None,
        )

    # Materialise inputs with seed=0 — identical for golden and SUT.
    numpy_inputs = materialize_inputs(test.inputs, seed=0)

    # --- golden ---
    try:
        golden_outputs: List[torch.Tensor] = config.run_golden(
            model_proto, numpy_inputs
        )
    except Exception:
        return TestResult(
            unique_name=test.unique_name,
            compilation_error=None,
            runtime_error="Golden evaluation failed:\n" + traceback.format_exc(),
            trace=None,
            golden_trace=None,
        )

    # --- SUT (compile + run) ---
    try:
        sut_outputs: List[torch.Tensor] = config.run_backend(model_proto, numpy_inputs)
    except Exception:
        tb = traceback.format_exc()
        # Heuristic: distinguish compile vs runtime failures by the presence of
        # typical compilation exception text.  A false-positive here is
        # harmless — it just determines which error field is populated.
        if "compilation" in tb.lower() or "lowering" in tb.lower():
            return TestResult(
                unique_name=test.unique_name,
                compilation_error=tb,
                runtime_error=None,
                trace=None,
                golden_trace=None,
            )
        return TestResult(
            unique_name=test.unique_name,
            compilation_error=None,
            runtime_error=tb,
            trace=None,
            golden_trace=None,
        )

    # Build trace / golden_trace — one TraceItem per call (just one here).
    # inputs are the numpy arrays converted to torch tensors so ValueReport
    # can compare them if needed; they should always match since it's the same
    # seed.
    torch_inputs = [torch.from_numpy(a) for a in numpy_inputs]

    # A mismatched output count would make the compare below silently align
    # the wrong tensors; surface it as a runtime error instead.
    if len(sut_outputs) != len(golden_outputs):
        return TestResult(
            unique_name=test.unique_name,
            compilation_error=None,
            runtime_error=(
                f"Output count mismatch: SUT returned {len(sut_outputs)}, "
                f"golden returned {len(golden_outputs)}."
            ),
            trace=None,
            golden_trace=None,
        )

    # Wrap multiple outputs as a tuple (single output stays as-is).
    if len(sut_outputs) == 1:
        sut_out = sut_outputs[0]
        golden_out = golden_outputs[0]
    else:
        sut_out = tuple(sut_outputs)
        golden_out = tuple(golden_outputs)

    trace = [TraceItem(symbol=_SYMBOL, inputs=torch_inputs, output=sut_out)]
    golden_trace = [TraceItem(symbol=_SYMBOL, inputs=torch_inputs, output=golden_out)]

    return TestResult(
        unique_name=test.unique_name,
        compilation_error=None,
        runtime_error=None,
        trace=trace,
        golden_trace=golden_trace,
    )


def run_tests(
    tests,
    config: OnnxLinalgTestConfig,
    verbose: bool = False,
) -> List[TestResult]:
    results = []
    for test in tests:
        if verbose:
            print(f'  Running "{test.unique_name}"...')
        result = _run_one(test, config)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ONNX-native end-to-end tests.")
    parser.add_argument(
        "-c",
        "--config",
        choices=["linalg", "tosa"],
        default="linalg",
        help="Backend to lower to and execute (default: linalg).",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default=".*",
        help="Regular expression to select tests by unique_name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Report test results with additional detail.",
    )
    parser.add_argument(
        "--ignore_failures",
        default=False,
        action="store_true",
        help="Return exit code 0 even if tests fail (unblocks pipelines).",
    )
    return parser


def main():
    args = _get_argparse().parse_args()

    # Resolve config.
    if args.config == "linalg":
        config = OnnxLinalgTestConfig()
        xfail_set = ONNX_E2E_LINALG_XFAIL_SET
        crashing_set = ONNX_E2E_LINALG_CRASHING_SET
    elif args.config == "tosa":
        config = OnnxTosaBackendTestConfig()
        xfail_set = ONNX_E2E_TOSA_XFAIL_SET
        crashing_set = ONNX_E2E_TOSA_CRASHING_SET
    else:
        print(f"ERROR: unknown config {args.config!r}", file=sys.stderr)
        sys.exit(1)

    # Skip known crashers.
    available_tests = [
        t for t in GLOBAL_ONNX_TEST_REGISTRY if t.unique_name not in crashing_set
    ]

    # Apply filter.
    tests = [t for t in available_tests if re.match(args.filter, t.unique_name)]
    if not tests:
        print(f"ERROR: filter {args.filter!r} matched no tests. Available tests:")
        for t in available_tests:
            print(f"  {t.unique_name}")
        sys.exit(1)

    # Run.
    results = run_tests(tests, config, verbose=args.verbose)

    # Report — reuses torch_mlir_e2e_test.reporting.report_results unchanged.
    failed = report_results(
        results,
        expected_failures=xfail_set,
        verbose=args.verbose,
        config=args.config,
    )

    if args.ignore_failures:
        sys.exit(0)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
