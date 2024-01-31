# RUN: %PYTHON -s %s 2>&1 | FileCheck %s

import gc
import sys
import torch
from torch_mlir import torchscript


def run_test(f):
    print("TEST:", f.__name__, file=sys.stderr)
    f()
    gc.collect()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear = torch.nn.Linear(20, 30)

    def forward(self, x):
        x = self.linear(x)
        return x


# CHECK-LABEL: TEST: test_enable_ir_printing
@run_test
def test_enable_ir_printing():
    torchscript.compile(TinyModel(),
                       torch.ones(1, 3, 20, 20),
                       output_type="linalg-on-tensors",
                       enable_ir_printing=True)
# CHECK: // -----// IR Dump Before Canonicalizer (canonicalize)
# CHECK-NEXT: module attributes {torch.debug_module_name = "TinyModel"} {
