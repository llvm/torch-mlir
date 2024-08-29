# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

# Helpers for the other tests.

import torch
from torch._C import CompilationUnit

# RUN: %PYTHON %s


# Import TorchScript IR string as ScriptFunction.
def create_script_function(func_name, ts_ir_str, **kwargs):
    cu = CompilationUnit()
    return cu.create_function(func_name, torch._C.parse_ir(ts_ir_str, **kwargs))
