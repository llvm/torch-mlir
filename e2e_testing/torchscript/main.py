#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pickle
import re
import sys

from npcomp_torchscript.e2e_test.framework import run_tests
from npcomp_torchscript.e2e_test.reporting import report_results
from npcomp_torchscript.e2e_test.registry import GLOBAL_TEST_REGISTRY

# Available test configs.
from npcomp_torchscript_e2e_test_configs import (
    NpcompBackendTestConfig, NativeTorchTestConfig, TorchScriptTestConfig
)

from npcomp.compiler.pytorch.backend import is_iree_enabled
IREE_ENABLED = is_iree_enabled()
if IREE_ENABLED:
    from npcomp.compiler.pytorch.backend.iree import IreeNpcompBackend
from npcomp.compiler.pytorch.backend.refbackend import RefBackendNpcompBackend

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
from . import list_programs
from . import reduction

def _get_argparse():
    config_choices = ['native_torch', 'torchscript', 'refbackend']
    if IREE_ENABLED:
        config_choices += ['iree']
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('-c', '--config',
        choices=config_choices,
        default='refbackend',
        help=f'''
Meaning of options:
"refbackend": run through npcomp's RefBackend.
"iree"{'' if IREE_ENABLED else '(disabled)'}: run through npcomp's IREE backend.
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
npcomp).
See `build_tools/torchscript_e2e_heavydep_tests/generate_serialized_tests.sh`
for more information on building these artifacts.
''')
    return parser

def main():
    args = _get_argparse().parse_args()

    # Find the selected config.
    if args.config == 'refbackend':
        config = NpcompBackendTestConfig(RefBackendNpcompBackend())
    elif args.config == 'iree':
        config = NpcompBackendTestConfig(IreeNpcompBackend())
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
