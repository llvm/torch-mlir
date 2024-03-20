import os
from typing import Tuple

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
            enable_ir_printing=True,
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

# @run
# def test_index_add():
#     class Index_add(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, x:Tensor, dim:torch.int, index:Tensor, source:Tensor, alpha=1) -> Tensor:
#             return torch.ops.aten.index_add(x, dim, index, source)
    
#     index_add = Transform(
#         Index_add(), 
#         torch.randn(128, 128, dtype=torch.float), 
#         0,
#         torch.tensor([8, 16, 32, 64, 127, 48, 72, 96], dtype=torch.int),
#         torch.randn(8, 128, dtype=torch.float),
#         )
#     index_add.run()

# @run
def test_sigmoid():
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor) -> Tensor:
            return torch.sigmoid(x)

    sigmoid = Transform(Sigmoid(), torch.randn(128, 128))
    sigmoid.run()

# @run
# TODO aten to linalg error
def test_sigmoid_backward():
    class Sigmoid_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad:Tensor, input:Tensor) -> Tensor:
            return torch.ops.aten.sigmoid_backward(grad, input)
    sigmoid_backward = Transform(Sigmoid_backward(), torch.randn(128, 128), torch.randn(128, 128))
    sigmoid_backward.run()

# # @run
# def test_logit_backward():
#     class Logit_backward(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, grad:Tensor, input:Tensor) -> Tensor:
#             return torch.ops.aten.logit_backward(grad, input)
#     logit_backward = Transform(Logit_backward(), torch.randn(128, 128), torch.randn(128, 128))
#     logit_backward.run()

# @run
def test_tanh_backward():
    class Tanh_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad:Tensor, input:Tensor) -> Tensor:
            return torch.ops.aten.tanh_backward(grad, input)
    tanh_backward = Transform(Tanh_backward(), torch.randn(128, 128), torch.randn(128, 128))
    tanh_backward.run()

# @run
def test_avg_pool2d():
    class Avg_pool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor, kernel_size:Tuple[int, ...]) -> Tensor:
            return torch.nn.functional.avg_pool2d(x, kernel_size)

    avg_pool2d = Transform(Avg_pool2d(), torch.randn(1, 1, 128, 128), (2, 2))
    avg_pool2d.run()

# @run
# aten to linalg error
def test_adaptive_avg_pool2d_backward():
    class Adaptive_avg_pool2d_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad:Tensor, input:Tensor) -> Tensor:
            return torch.ops.aten._adaptive_avg_pool2d_backward(grad, input)
    adaptive_avg_pool2d_backward = Transform(Adaptive_avg_pool2d_backward(), torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128))
    adaptive_avg_pool2d_backward.run()


# @run
# def test_softplus_backward():
#     class Softplus_backward(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, grad:Tensor, input:Tensor) -> Tensor:
#             return torch.ops.aten.softplus_backward(grad, input, beta=1, threshold=20)

#     softplus_backward = Transform(Softplus_backward(), torch.randn(128, 128), torch.randn(128, 128), Tensor(1), Tensor(20))
#     softplus_backward.run()

# @run
def test_log_sigmoid_forward():
    class Log_sigmoid_forward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input:Tensor) -> Tensor:
            return torch.nn.functional.logsigmoid(input)

    log_sigmoid_forward = Transform(Log_sigmoid_forward(), torch.randn(128, 128))
    log_sigmoid_forward.run()

# @run
def test_softmax():
    class Softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor, dim:int) -> Tensor:
            return torch.nn.functional.softmax(x, dim, torch.float32)
    softmax = Transform(Softmax(), torch.randn(128, 128), 1)
    softmax.run()

# @run
def test_leaky_relu_backward():
    class Leaky_relu_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad:Tensor, input:Tensor):
            return torch.ops.aten.leaky_relu_backward(grad, input, negative_slope=0.1, self_is_result=False)

    leaky_relu_backward = Transform(Leaky_relu_backward(), torch.randn(128, 128), torch.randn(128, 128))
    leaky_relu_backward.run()

# @run
def test_leaky_relu():
    class Leaky_relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor) -> Tensor:
            return torch.nn.functional.softmax(x)
    leaky_relu = Transform(Leaky_relu(), torch.randn(128, 128))
    leaky_relu.run()

# @run
def test_glu():
    class Glu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,input):
            return torch.nn.functional.glu(input, dim=-1)
    glu = Transform(Glu(), torch.randn(128, 128))
    glu.run()

# @run
def test_elu():
    class Elu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,input):
            return torch.nn.functional.elu(input, alpha=1.0, inplace=False)
    elu = Transform(Elu(), torch.randn(128, 128))
    elu.run()

@run
def test_smoothl1loss():
    class Smooth_l1_loss(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, target):
            return torch.nn.functional.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='none',beta=1.0)
    smooth_l1_loss = Transform(Smooth_l1_loss(), torch.randn(128, 128), torch.randn(128, 128))
    smooth_l1_loss.run()

# @run
def test_logical_not():
    class Logicalnot(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor):
            return torch.logical_not(x)
    softmax = Transform(Logicalnot(), torch.randn(128, 128))
    softmax.run()

# @run
def test_transpose():
    class Transpose(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:Tensor, dim0, dim1):
            return torch.transpose(x, dim0, dim1)
    transpose = Transform(Transpose(), torch.randn(128, 128), 0, 1)
    transpose.run()

# @run
def test_cumsum():
    class Cumsum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, dim):
            return torch.cumsum(x, dim)
    cumsum = Transform(Cumsum(), torch.randn(1024), 0)
    cumsum.run()
