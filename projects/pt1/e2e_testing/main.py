# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse
import re
import sys

import torch

torch.device("cpu")

from torch_mlir_e2e_test.framework import run_tests
from torch_mlir_e2e_test.reporting import report_results
from torch_mlir_e2e_test.registry import GLOBAL_TEST_REGISTRY


# Available test configs.
from torch_mlir_e2e_test.configs import (
    LazyTensorCoreTestConfig,
    NativeTorchTestConfig,
    OnnxBackendTestConfig,
    TorchScriptTestConfig,
    TorchDynamoTestConfig,
    JITImporterTestConfig,
    FxImporterTestConfig,
)

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import (
    LinalgOnTensorsTosaBackend,
)
from torch_mlir_e2e_test.stablehlo_backends.linalg_on_tensors import (
    LinalgOnTensorsStablehloBackend,
)

from .xfail_sets import (
    LINALG_XFAIL_SET,
    LINALG_CRASHING_SET,
    STABLEHLO_PASS_SET,
    STABLEHLO_CRASHING_SET,
    TOSA_PASS_SET,
    TOSA_CRASHING_SET,
    LTC_XFAIL_SET,
    LTC_CRASHING_SET,
    TORCHDYNAMO_XFAIL_SET,
    TORCHDYNAMO_CRASHING_SET,
    ONNX_CRASHING_SET,
    ONNX_XFAIL_SET,
    FX_IMPORTER_XFAIL_SET,
    FX_IMPORTER_CRASHING_SET,
    FX_IMPORTER_STABLEHLO_XFAIL_SET,
    FX_IMPORTER_STABLEHLO_CRASHING_SET,
    FX_IMPORTER_TOSA_CRASHING_SET,
    FX_IMPORTER_TOSA_XFAIL_SET,
    ONNX_TOSA_XFAIL_SET,
    ONNX_TOSA_CRASHING_SET,
)

# Import tests to register them in the global registry.
from torch_mlir_e2e_test.test_suite import register_all_tests

register_all_tests()


def _get_argparse():
    config_choices = [
        "native_torch",
        "torchscript",
        "linalg",
        "stablehlo",
        "tosa",
        "lazy_tensor_core",
        "torchdynamo",
        "onnx",
        "onnx_tosa",
        "fx_importer",
        "fx_importer_stablehlo",
        "fx_importer_tosa",
    ]
    parser = argparse.ArgumentParser(description="Run torchscript e2e tests.")
    parser.add_argument(
        "-c",
        "--config",
        choices=config_choices,
        default="linalg",
        help=f"""
Meaning of options:
"linalg": run through torch-mlir"s default Linalg-on-Tensors backend.
"tosa": run through torch-mlir"s default TOSA backend.
"stablehlo": run through torch-mlir"s default Stablehlo backend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic; ALL tests should pass in this configuration).
"torchscript": compile the model to a torch.jit.ScriptModule, and then run that as-is (useful for verifying TorchScript is modeling the program correctly).
"lazy_tensor_core": run the model through the Lazy Tensor Core frontend and execute the traced graph.
"torchdynamo": run the model through the TorchDynamo frontend and execute the graph using Linalg-on-Tensors.
"onnx": export to the model via onnx and reimport using the torch-onnx-to-torch path.
"fx_importer": run the model through the fx importer frontend and execute the graph using Linalg-on-Tensors.
"fx_importer_stablehlo": run the model through the fx importer frontend and execute the graph using Stablehlo backend.
"fx_importer_tosa": run the model through the fx importer frontend and execute the graph using the TOSA backend.
"onnx_tosa": Import ONNX to Torch via the torch-onnx-to-torch path and execute the graph using the TOSA backend.
""",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default=".*",
        help="""
Regular expression specifying which tests to include in this run.
""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="report test results with additional detail",
    )
    parser.add_argument(
        "-s",
        "--sequential",
        default=False,
        action="store_true",
        help="""Run tests sequentially rather than in parallel.
This can be useful for debugging, since it runs the tests in the same process,
which make it easier to attach a debugger or get a stack trace.""",
    )
    parser.add_argument(
        "--crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed",
        metavar="TEST",
        type=str,
        nargs="+",
        help="A set of tests to not attempt to run, since they crash and cannot be XFAILed.",
    )
    parser.add_argument(
        "--ignore_failures",
        default=False,
        action="store_true",
        help="return exit code 0 even if the test fails to unblock pipeline",
    )
    return parser


def main():
    args = _get_argparse().parse_args()

    all_test_unique_names = set(test.unique_name for test in GLOBAL_TEST_REGISTRY)

    # Find the selected config.
    if args.config == "linalg":
        config = JITImporterTestConfig(RefBackendLinalgOnTensorsBackend())
        xfail_set = LINALG_XFAIL_SET
        crashing_set = LINALG_CRASHING_SET
    elif args.config == "stablehlo":
        config = JITImporterTestConfig(LinalgOnTensorsStablehloBackend(), "stablehlo")
        xfail_set = all_test_unique_names - STABLEHLO_PASS_SET
        crashing_set = STABLEHLO_CRASHING_SET
    elif args.config == "tosa":
        config = JITImporterTestConfig(LinalgOnTensorsTosaBackend(), "tosa")
        xfail_set = all_test_unique_names - TOSA_PASS_SET
        crashing_set = TOSA_CRASHING_SET
    elif args.config == "native_torch":
        config = NativeTorchTestConfig()
        xfail_set = set()
        crashing_set = set()
    elif args.config == "torchscript":
        config = TorchScriptTestConfig()
        xfail_set = set()
        crashing_set = set()
    elif args.config == "lazy_tensor_core":
        config = LazyTensorCoreTestConfig()
        xfail_set = LTC_XFAIL_SET
        crashing_set = LTC_CRASHING_SET
    elif args.config == "fx_importer":
        config = FxImporterTestConfig(RefBackendLinalgOnTensorsBackend())
        xfail_set = FX_IMPORTER_XFAIL_SET
        crashing_set = FX_IMPORTER_CRASHING_SET
    elif args.config == "fx_importer_stablehlo":
        config = FxImporterTestConfig(LinalgOnTensorsStablehloBackend(), "stablehlo")
        xfail_set = FX_IMPORTER_STABLEHLO_XFAIL_SET
        crashing_set = FX_IMPORTER_STABLEHLO_CRASHING_SET
    elif args.config == "fx_importer_tosa":
        config = FxImporterTestConfig(LinalgOnTensorsTosaBackend(), "tosa")
        xfail_set = FX_IMPORTER_TOSA_XFAIL_SET
        crashing_set = FX_IMPORTER_TOSA_CRASHING_SET
    elif args.config == "torchdynamo":
        # TODO: Enanble runtime verification and extend crashing set.
        config = TorchDynamoTestConfig(
            RefBackendLinalgOnTensorsBackend(generate_runtime_verification=False)
        )
        xfail_set = TORCHDYNAMO_XFAIL_SET
        crashing_set = TORCHDYNAMO_CRASHING_SET
    elif args.config == "onnx":
        config = OnnxBackendTestConfig(RefBackendLinalgOnTensorsBackend())
        xfail_set = ONNX_XFAIL_SET
        crashing_set = ONNX_CRASHING_SET
    elif args.config == "onnx_tosa":
        config = OnnxBackendTestConfig(LinalgOnTensorsTosaBackend(), output_type="tosa")
        xfail_set = ONNX_TOSA_XFAIL_SET
        crashing_set = ONNX_TOSA_CRASHING_SET

    do_not_attempt = set(
        args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed or []
    ).union(crashing_set)
    available_tests = [
        test for test in GLOBAL_TEST_REGISTRY if test.unique_name not in do_not_attempt
    ]
    if args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed is not None:
        for arg in args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed:
            if arg not in all_test_unique_names:
                print(
                    f"ERROR: --crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed argument '{arg}' is not a valid test name"
                )
                sys.exit(1)

    # Find the selected tests, and emit a diagnostic if none are found.
    tests = [
        test for test in available_tests if re.match(args.filter, test.unique_name)
    ]
    if len(tests) == 0:
        print(f"ERROR: the provided filter {args.filter!r} does not match any tests")
        print("The available tests are:")
        for test in available_tests:
            print(test.unique_name)
        sys.exit(1)

    # Run the tests.
    results = run_tests(tests, config, args.sequential, args.verbose)

    # Report the test results.
    failed = report_results(results, xfail_set, args.verbose, args.config)
    if args.config == "torchdynamo":
        print(
            "\033[91mWarning: the TorchScript based dynamo support is deprecated. "
            "The config for torchdynamo is planned to be removed in the future.\033[0m"
        )
    if args.ignore_failures:
        sys.exit(0)
    sys.exit(1 if failed else 0)


def _suppress_warnings():
    import warnings

    # Ignore warning due to Python bug:
    # https://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array
    warnings.filterwarnings(
        "ignore",
        message="A builtin ctypes object gave a PEP3118 format string that does not match its itemsize",
    )


if __name__ == "__main__":
    _suppress_warnings()
    main()
