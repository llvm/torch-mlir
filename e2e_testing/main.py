# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse
import re
import sys

from torch_mlir_e2e_test.framework import run_tests
from torch_mlir_e2e_test.reporting import report_results
from torch_mlir_e2e_test.registry import GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.serialization import deserialize_all_tests_from


# Available test configs.
from torch_mlir_e2e_test.configs import (
    LazyTensorCoreTestConfig,
    LinalgOnTensorsBackendTestConfig,
    MhloBackendTestConfig,
    NativeTorchTestConfig,
    TorchScriptTestConfig,
    TosaBackendTestConfig,
    EagerModeTestConfig
)

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend
from torch_mlir_e2e_test.mhlo_backends.linalg_on_tensors import LinalgOnTensorsMhloBackend
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend

from .xfail_sets import REFBACKEND_XFAIL_SET, MHLO_PASS_SET, TOSA_PASS_SET, EAGER_MODE_XFAIL_SET, LTC_XFAIL_SET

# Import tests to register them in the global registry.
from torch_mlir_e2e_test.test_suite import register_all_tests
register_all_tests()

def _get_argparse():
    config_choices = ['native_torch', 'torchscript', 'refbackend', 'mhlo', 'tosa', 'eager_mode', 'lazy_tensor_core']
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('-c', '--config',
        choices=config_choices,
        default='refbackend',
        help=f'''
Meaning of options:
"refbackend": run through torch-mlir's RefBackend.
"mhlo": run through torch-mlir's default MHLO backend.
"tosa": run through torch-mlir's default TOSA backend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic; ALL tests should pass in this configuration).
"torchscript": compile the model to a torch.jit.ScriptModule, and then run that as-is (useful for verifying TorchScript is modeling the program correctly).
"eager_mode": run through torch-mlir's eager mode frontend, using RefBackend for execution.
"lazy_tensor_core": run the model through the Lazy Tensor Core frontend and execute the traced graph. 
''')
    parser.add_argument('-f', '--filter', default='.*', help='''
Regular expression specifying which tests to include in this run.
''')
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action='store_true',
                        help='report test results with additional detail')
    parser.add_argument('--serialized-test-dir', default=None, type=str, help='''
The directory containing serialized pre-built tests.
Right now, these are additional tests which require heavy Python dependencies
to generate (or cannot even be generated with the version of PyTorch used by
torch-mlir).
See `build_tools/e2e_heavydep_tests/generate_serialized_tests.sh`
for more information on building these artifacts.
''')
    parser.add_argument('-s', '--sequential',
                        default=False,
                        action='store_true',
                        help='''Run tests sequentially rather than in parallel.
This can be useful for debugging, since it runs the tests in the same process,
which make it easier to attach a debugger or get a stack trace.''')
    parser.add_argument('--crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed',
                        metavar="TEST", type=str, nargs='+',
                        help='A set of tests to not attempt to run, since they crash and cannot be XFAILed.')
    return parser

def main():
    args = _get_argparse().parse_args()

    if args.serialized_test_dir:
        deserialize_all_tests_from(args.serialized_test_dir)
    all_test_unique_names = set(
        test.unique_name for test in GLOBAL_TEST_REGISTRY)

    # Find the selected config.
    if args.config == 'refbackend':
        config = LinalgOnTensorsBackendTestConfig(RefBackendLinalgOnTensorsBackend())
        xfail_set = REFBACKEND_XFAIL_SET
    if args.config == 'tosa':
        config = TosaBackendTestConfig(LinalgOnTensorsTosaBackend())
        xfail_set = all_test_unique_names - TOSA_PASS_SET
    if args.config == 'mhlo':
        config = MhloBackendTestConfig(LinalgOnTensorsMhloBackend())
        xfail_set = all_test_unique_names - MHLO_PASS_SET
    elif args.config == 'native_torch':
        config = NativeTorchTestConfig()
        xfail_set = {}
    elif args.config == 'torchscript':
        config = TorchScriptTestConfig()
        xfail_set = {}
    elif args.config == 'eager_mode':
        config = EagerModeTestConfig()
        xfail_set = EAGER_MODE_XFAIL_SET
    elif args.config == 'lazy_tensor_core':
        config = LazyTensorCoreTestConfig()
        xfail_set = LTC_XFAIL_SET

    do_not_attempt = set(args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed or [])
    available_tests = [test for test in GLOBAL_TEST_REGISTRY if test.unique_name not in do_not_attempt]
    if args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed is not None:
        for arg in args.crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed:
            if arg not in all_test_unique_names:
                print(f'ERROR: --crashing_tests_to_not_attempt_to_run_and_a_bug_is_filed argument "{arg}" is not a valid test name')
                sys.exit(1)

    # Find the selected tests, and emit a diagnostic if none are found.
    tests = [
        test for test in available_tests
        if re.match(args.filter, test.unique_name)
    ]
    if len(tests) == 0:
        print(
            f'ERROR: the provided filter {args.filter!r} does not match any tests'
        )
        print('The available tests are:')
        for test in available_tests:
            print(test.unique_name)
        sys.exit(1)

    # Run the tests.
    results = run_tests(tests, config, args.sequential, args.verbose)

    # Report the test results.
    failed = report_results(results, xfail_set, args.verbose)
    sys.exit(1 if failed else 0)

def _suppress_warnings():
    import warnings
    # Ignore warning due to Python bug:
    # https://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array
    warnings.filterwarnings("ignore",
                            message="A builtin ctypes object gave a PEP3118 format string that does not match its itemsize")

if __name__ == '__main__':
    _suppress_warnings()
    main()
