import os
from torch_mlir import fx
import torch

from torch_mlir.compiler_utils import run_pipeline_with_repro_report


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()

def save_mlir(module, name, ir):
    module_strs = str(module)
    mlir_name = name + ".mlir"
    cwd = os.getcwd()
    mlir_path = os.path.join(cwd,"iree_test", ir)
    if not os.path.exists(mlir_path):
        os.makedirs(mlir_path)
    with open(os.path.join(mlir_path, mlir_name), 'w') as f:
        f.write(module_strs)

class Transform:
    def __init__(self, f, args):
        self.f = f
        self.kernel_name = f.__class__.__name__
        self.args = args
        self.module = None
    
    def get_torchIR(self):
        self.module = fx.export_and_import(self.f, self.args)
        save_mlir(self.module, self.kernel_name, "torch-aten")

    def lower_linalg(self):
        run_pipeline_with_repro_report(
            self.module,
            (
                "builtin.module("
                "func.func(torch-decompose-complex-ops),"
                "torch-backend-to-linalg-on-tensors-backend-pipeline)"
            ),
            "Lowering TorchFX IR -> Linalg IR",
            enable_ir_printing=False,
        )
        save_mlir(self.module, self.kernel_name, "linalg-ir")
    
    def run(self):
        self.get_torchIR()
        self.lower_linalg()
        return self.module

################################################################
#  Add torch kernel example
################################################################
@run
def test_sigmoid():
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    sigmoid = Transform(Sigmoid(), torch.randn(8, 128))
    module = sigmoid.run()

@run
def test_softmax():
    class Softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.softmax(x, -1, torch.float32)
    softmax = Transform(Softmax(), torch.randn(8, 128))
    module = softmax.run()
    # print("====================================")
    # print(module)
