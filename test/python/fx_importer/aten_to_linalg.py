import os
from torch_mlir import fx
import torch
from torch import Tensor

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
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.kernel_name = f.__class__.__name__
        self.args = args
        self.kwargs = kwargs
        self.module = None
    
    def get_torchIR(self):
        self.module = fx.export_and_import(self.f, *self.args, **self.kwargs)
        print("aten ir:")
        print(self.module)
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
        print("linalg ir:")
        print(self.module)
        save_mlir(self.module, self.kernel_name, "linalg-ir")
    
    def run(self):
        self.get_torchIR()
        self.lower_linalg()
        # return self.module

################################################################
#  Add torch kernel example
################################################################
        
# error TODO
# python exception: Failure while executing pass pipeline:
# error: unknown: failed to legalize operation 'torch.constant.int'
# note: unknown: see current operation: %0 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
# @run
def test_index_add():
    class Index_add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor, dim, index:Tensor, source:Tensor) -> Tensor:
            return x.index_add_(dim, index, source)
    
    index_add = Transform(
        Index_add(), 
        torch.randn(128, 128, dtype=torch.float), 
        0,
        torch.tensor([8, 16, 32, 64, 127, 48, 72, 96], dtype=torch.int),
        torch.randn(8, 128, dtype=torch.float),
        )
    index_add.run()

@run
def test_sigmoid():
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor) -> Tensor:
            return torch.sigmoid(x)

    sigmoid = Transform(Sigmoid(), torch.randn(128, 128))
    sigmoid.run()

@run
def test_softmax():
    class Softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor, dim:int) -> Tensor:
            return torch.nn.functional.softmax(x, dim, torch.float32)
    softmax = Transform(Softmax(), torch.randn(128, 128), 1)
    softmax.run()

