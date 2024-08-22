# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import enum

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder


class Color(enum.Enum):
    RED = 1
    GREEN = 2


# RUN: %PYTHON %s

mb = ModuleBuilder()

# To test errors, use a type that we don't support yet.
try:

    @mb.import_function
    @torch.jit.script
    def import_class(x: Color):
        return x

except Exception as e:
    # TODO: Once diagnostics are enabled, verify the actual error emitted.
    assert str(e) == "unsupported type in function schema: 'Enum<__torch__.Color>'"
else:
    assert False, "Expected exception"
