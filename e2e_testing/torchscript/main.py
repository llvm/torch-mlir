#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pickle
import re
import sys

from torch_mlir_e2e_test.torchscript.framework import run_tests
from torch_mlir_e2e_test.torchscript.reporting import report_results
from torch_mlir_e2e_test.torchscript.registry import GLOBAL_TEST_REGISTRY

# Available test configs.
from torch_mlir_e2e_test.torchscript.configs import (
    LinalgOnTensorsBackendTestConfig, NativeTorchTestConfig, TorchScriptTestConfig
)

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .xfail_sets import XFAIL_SETS

# Import tests to register them in the global registry.
# Make sure to use `tools/torchscript_e2e_test.sh` wrapper for invoking
# this script.
from . import basic
from . import vision_models
from . import mlp
from . import conv
from . import batchnorm
from . import quantized_models
from . import elementwise
from . import reduction

def _get_argparse():
    # TODO: Allow pulling in an out-of-tree backend, so downstream can easily
    # plug into the e2e tests.
    config_choices = ['native_torch', 'torchscript', 'refbackend']
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('-c', '--config',
        choices=config_choices,
        default='refbackend',
        help=f'''
Meaning of options:
"refbackend": run through torch-mlir's RefBackend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic; ALL tests should pass in this configuration).
"torchscript": compile the model to a torch.jit.ScriptModule, and then run that as-is (useful for verifying TorchScript is modeling the program correctly).
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
See `build_tools/torchscript_e2e_heavydep_tests/generate_serialized_tests.sh`
for more information on building these artifacts.
''')
    return parser

def main():
    args = _get_argparse().parse_args()

    # Find the selected config.
    if args.config == 'refbackend':
        config = LinalgOnTensorsBackendTestConfig(RefBackendLinalgOnTensorsBackend())
    elif args.config == 'native_torch':
        config = NativeTorchTestConfig()
    elif args.config == 'torchscript':
        config = TorchScriptTestConfig()

    all_tests = list(GLOBAL_TEST_REGISTRY)
    if args.serialized_test_dir:
        for root, dirs, files in os.walk(args.serialized_test_dir):
            for filename in files:
                with open(os.path.join(root, filename), 'rb') as f:
                    all_tests.append(pickle.load(f).as_test())

    # Find the selected tests, and emit a diagnostic if none are found.
    tests = [
        test for test in all_tests
        if re.match(args.filter, test.unique_name)
    ]
    if len(tests) == 0:
        print(
            f'ERROR: the provided filter {args.filter!r} does not match any tests'
        )
        print('The available tests are:')
        for test in all_tests:
            print(test.unique_name)
        sys.exit(1)

    # Run the tests.
    results = run_tests(tests, config)

    # Report the test results.
    failed = report_results(results, XFAIL_SETS[args.config], args.verbose)
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
