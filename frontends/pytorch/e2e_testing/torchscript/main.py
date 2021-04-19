#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

from torch_mlir.torchscript.e2e_test.framework import run_tests, report_results
from torch_mlir.torchscript.e2e_test.registry import GLOBAL_TEST_REGISTRY

# Available test configs.
from torch_mlir.torchscript.e2e_test.configs import (
    RefBackendTestConfig, NativeTorchTestConfig, TorchScriptTestConfig
)

# Import tests to register them in the global registry.
import basic

def main():
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('--config',
        choices=['native_torch', 'torchscript', 'refbackend'],
        default='refbackend',
        help='''
Meaning of options:
"refbackend": run through npcomp's RefBackend.
"native_torch": run the torch.nn.Module as-is without compiling (useful for verifying model is deterministic).
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
