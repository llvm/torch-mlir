#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

from torch_mlir.torchscript.e2e_test.framework import run_tests
from torch_mlir.torchscript.e2e_test.reporting import report_results
from torch_mlir.torchscript.e2e_test.registry import GLOBAL_TEST_REGISTRY

# Available test configs.
from torch_mlir.torchscript.e2e_test.configs import (
    RefBackendTestConfig, NativeTorchTestConfig, TorchScriptTestConfig
)

# Import tests to register them in the global registry.
# TODO: Use a relative import.
# That requires invoking this file as a "package" though, which makes it
# not possible to just do `python main.py`. Instead, it requires something
# like `python -m tochscript.main` which is annoying because it can only
# be run from a specific directory.
# TODO: Find out best practices for python "main" files.
import basic
import vision_models

def main():
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('--config',
        choices=['native_torch', 'torchscript', 'refbackend'],
        default='refbackend',
        help='''
Meaning of options:
"refbackend": run through npcomp's RefBackend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic; ALL tests should pass in this configuration).
"torchscript": compile the model to a torch.jit.ScriptModule, and then run that as-is (useful for verifying TorchScript is modeling the program correctly).
''')
    args = parser.parse_args()
    if args.config == 'refbackend':
        config = RefBackendTestConfig()
    elif args.config == 'native_torch':
        config = NativeTorchTestConfig()
    elif args.config == 'torchscript':
        config = TorchScriptTestConfig()
    results = run_tests(GLOBAL_TEST_REGISTRY, config)
    report_results(results)

if __name__ == '__main__':
    main()
