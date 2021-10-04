# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse
import os
import pickle
import re
import sys

from torch_mlir_e2e_test.torchscript.framework import TestConfig, run_tests
from torch_mlir_e2e_test.torchscript.reporting import report_results
from torch_mlir_e2e_test.torchscript.registry import GLOBAL_TEST_REGISTRY

# Available test configs.
from torch_mlir_e2e_test.torchscript.configs import (
    LinalgOnTensorsBackendTestConfig, NativeTorchTestConfig, TorchScriptTestConfig
)

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .xfail_sets import XFAIL_SETS, COMMON_TORCH_MLIR_LOWERING_XFAILS

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
    config_choices = ['native_torch', 'torchscript', 'refbackend', 'external']
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('-c', '--config',
        choices=config_choices,
        default='refbackend',
        help=f'''
Meaning of options:
"refbackend": run through torch-mlir's RefBackend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic; ALL tests should pass in this configuration).
"torchscript": compile the model to a torch.jit.ScriptModule, and then run that as-is (useful for verifying TorchScript is modeling the program correctly).
"external": use an external backend, specified by the `--external-backend` option.
''')
    parser.add_argument('--external-config',
        help=f'''
Specifies a path to a Python file, which will be `exec`'ed.
The file has the following contract:
- The global variable `config` should be set to an instance of `TestConfig`.
- `xfail_set` should be set to a set of test unique identifiers that are
  expected to fail. The global `COMMON_TORCH_MLIR_LOWERING_XFAILS` provides
  a common set of xfails that won't work on backends because torch-mlir
  itself does not handle them.
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
        xfail_set = XFAIL_SETS['refbackend']
    elif args.config == 'native_torch':
        config = NativeTorchTestConfig()
        xfail_set = XFAIL_SETS['native_torch']
    elif args.config == 'torchscript':
        config = TorchScriptTestConfig()
        xfail_set = XFAIL_SETS['torchscript']
    elif args.config == 'external':
        with open(args.external_config, 'r') as f:
            code = compile(f.read(), args.external_config, 'exec')
        exec_globals = {
            'COMMON_TORCH_MLIR_LOWERING_XFAILS': COMMON_TORCH_MLIR_LOWERING_XFAILS}
        exec(code, exec_globals)
        config = exec_globals.get('config')
        xfail_set = exec_globals.get('xfail_set')
        if config is None or not isinstance(config, TestConfig):
            print(
                f'ERROR: the script {args.external_config} did not set a global variable `config`'
            )
            sys.exit(1)
        if xfail_set is None:
            print(
                f'ERROR: the script {args.external_config} did not set a global variable `xfail_set`'
            )
            sys.exit(1)

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
    failed = report_results(results, xfail_set, args.verbose)
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
