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

# @run
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> all kernel aten to linalg

# @run
def test_permute():
    class Permute(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.permute(input, dim)
    permute = Transform(Permute(), torch.randn(128, 128), (1, 0))
    permute.run()

# @run
def test_tanh():
    class Tanh(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.tanh(input)
    tanh = Transform(Tanh(), torch.randn(128, 128))
    tanh.run()

# @run
def test_binary_cross_entropy():
    class Binary_cross_entropy(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, target):
            return torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
    binary_cross_entropy = Transform(Binary_cross_entropy(), torch.randn(128, 128, requires_grad=True), torch.rand(128, 128))
    binary_cross_entropy.run()

# @run
def test_layer_norm():
    class Layer_norm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, normalized_shap):
            return torch.nn.functional.layer_norm(input, normalized_shap, weight=None, bias=None, eps=1e-05)
    layer_norm = Transform(Layer_norm(), torch.randn(256, 128), (128,))
    layer_norm.run()

# @run 
def test_embedding_dense_backward():
    class Embedding_dense_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output_size, indices_size, num_weights, padding_idx, scale_grad_by_freq):
            return torch.ops.aten.embedding_dense_backward(grad_output_size, indices_size, num_weights, padding_idx, scale_grad_by_freq)
    embedding_dense_backward = Transform(Embedding_dense_backward(), torch.randn(128, 8, 1), torch.randn(128, 8), 32000, -1, True)
    embedding_dense_backward.run()

# @run
# def test_layer_norm():
#     class Layer_norm_backward(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, grad_out, input, normalized_shape, mean, rstd, ):
#             return torch.ops.aten.native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd,weight=None, bias=None)
#     layer_norm_backward = Transform(Layer_norm_backward(), torch.randn(256, 128), torch.randn(256, 128), (128,), torch.randn(128), torch.randn(128))
#     layer_norm_backward.run() #参数设置问题
    
# @run
def test_native_dropout():
    class Native_dropout(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
    native_dropout = Transform(Native_dropout(), torch.randn(4096, 53))
    native_dropout.run()

# @run
def test_tril():
    class Tril(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.tril(input, diagonal = 0)
    tril = Transform(Tril(), torch.randn(128, 128))
    tril.run()

# @run
def test_var_meancorrection():
    class Var_meancorrection(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.var_mean(input, dim=None, correction=1,keepdim=False)
    var_meancorrection = Transform(Var_meancorrection(), torch.randn(128, 128))
    var_meancorrection.run()

# @run
def test_where():
    class Where(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, condition, input, other):
            return torch.where(condition, input, other,out=None)
    alternating_2d_list = [[0 if i % 2 == 0 else 1 for i in range(128)] for j in range(128)]
    tensor = torch.tensor(alternating_2d_list)
    condition = tensor.to(torch.bool)
    where = Transform(Where(), condition, torch.randn(128, 128), torch.randn(128, 128))
    where.run()

# @run
def test_sliceTensor():
    class SliceTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, start, end, step):
            return torch.ops.aten.slice.Tensor(input, dim, start, end, step)
    sliceTensor = Transform(SliceTensor(), torch.randn(128, 128), 1, 0, 8, 4)
    sliceTensor.run()

# @run
def test_silu():
    class Silu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.silu(input, inplace=False)
    silu = Transform(Silu(), torch.randn(128, 128))
    silu.run()

# @run
def test_prod():
    class Prod(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.prod(input, dim, keepdim=False, dtype=None)
    prod = Transform(Prod(), torch.randn(128, 128), 1)
    prod.run()

# @run
def test_fulllike():
    class Fulllike(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, fill_value):
            return torch.full_like(input, fill_value)
    fulllike = Transform(Fulllike(), torch.ones(128, 128), 3.14)
    fulllike.run()
    
# @run
def test_full():
    class Full(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, size, fill_value):
            return torch.full(size, fill_value)
    full = Transform(Full(), (128, 128), 1.0)
    full.run()

# @run
def test_divScalar():
    class DivScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, scalar):
            return torch.div(input, scalar)
    divScalar = Transform(DivScalar(), torch.randn(128, 128), 3.14)
    divScalar.run()

# @run
def powTensor_Scalar():
    class PowTensor_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, exponent):
            return torch.pow(input, exponent)
    powTensor_Scalar = Transform(PowTensor_Scalar(), torch.randn(128, 128), 3)
    powTensor_Scalar.run()

# @run
def divTensor():
    class DivTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, tensor):
            return torch.div(input, tensor)
    divTensor = Transform(DivTensor(), torch.randn(128, 128), torch.randn(128, 128))
    divTensor.run()

# @run 
def index_select():
    class Index_select(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index):
            return torch.index_select(input, dim, index)
    index_select = Transform(Index_select(), torch.randn(128, 128), 0, torch.tensor([0, 8, 16, 24]))
    index_select.run()

# @run
# def index_select_out():
#     class Index_select_out(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, dim, index):
#             return torch.index_select(input, dim, index, output=torch.zeros(8, 128))
#     index_select_out = Transform(Index_select_out(), torch.randn(128, 128), 0, torch.tensor([0, 8, 16, 24, 32, 40, 48, 56]))
#     index_select_out.run() #参数设置问题

# @run
def masked_select():
    class Masked_select(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mask):
            return torch.masked_select(input, mask)
    alternating_2d_list = [[0 if i % 2 == 0 else 1 for i in range(128)] for j in range(128)]
    tensor = torch.tensor(alternating_2d_list)
    mask = tensor.to(torch.bool)
    masked_select = Transform(Masked_select(), torch.randn(128, 128), mask)
    masked_select.run() #aten to linalg error

# @run
def sumdim_IntList():
    class Sumdim_IntList(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.sum(input, dim, keepdim=False, dtype=None)
    sumdim_IntList = Transform(Sumdim_IntList(), torch.randn(128, 128), 0)
    sumdim_IntList.run()

# @run
def neg():
    class Neg(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.neg(input)
    neg = Transform(Neg(), torch.randn(128, 128))
    neg.run()

# @run
def rsqrt():
    class Rsqrt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.rsqrt(input)
    rsqrt = Transform(Rsqrt(), torch.randn(128, 128))
    rsqrt.run()

# @run
# def indexTensor():
#     class IndexTensor(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input):
#             return torch.ops.aten.index.Tensor(input, indices=torch.ones((1), dtype=torch.int64))
#     indexTensor = Transform(IndexTensor(), torch.randn(128, 128))
#     indexTensor.run() #参数设置问题
    
# @run
# def index_copy():
#     class Index_copy(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#         def forward(self, input, dim, index, source):
#             return torch.index_copy(input, dim, index, source)
#     index_copy = Transform(Index_copy(), torch.randn(128, 128), 0, torch.tensor([0, 4, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))
#     index_copy.run() #aten to linalg error
    
# @run
def meandim():
    class Meandim(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.mean(input, dim, keepdim=False, dtype=None, out=None)
    meandim = Transform(Meandim(), torch.randn(128, 128), 1)
    meandim.run()

# @run
# def unbindint():
#     class Unbindint(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, dim):
#             return torch.unbind(input, dim)
#     unbindint = Transform(Unbindint(), torch.tensor([[1, 2, 3],[4,5,6],[7,8,9]]), 0)
#     unbindint.run() #报错
    
# @run
# def convert_element_type():
#     class Convert_element_type(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, dtype):
#             return torch.prims.convert_element_type(input, dtype)
#     convert_element_type = Transform(Convert_element_type(), torch.randn(8,128), 1)
#     convert_element_type.run()  #forward函数中无法找到可以使用的return函数
    
# @run
def amax():
    class Amax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.amax(input, dim, keepdim=False)
    amax = Transform(Amax(), torch.randn(128, 128), (1,))
    amax.run()

# @run
def neScalar():
    class NeScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.ne(input, other)
    neScalar = Transform(NeScalar(), torch.randn(128, 128), 3.14)
    neScalar.run()

# @run
def gather():
    class Gather(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index):
            return torch.gather(input, dim, index, sparse_grad=False)
    gather = Transform(Gather(), torch.randn(3, 2), 1, torch.tensor([[0,1], [1,0], [0,0]]))
    gather.run()

# @run
def mmout():
    class Mmout(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mat):
            return torch.mm(input, mat)
    mmout = Transform(Mmout(), torch.randn(8, 128), torch.randn(128, 8))
    mmout.run()

# @run
def mul():
    class Mul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.mul(input, other)
    mul = Transform(Mul(), torch.randn(8, 128), torch.randn(8, 128))
    mul.run()

# @run
def exp():
    class Exp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.exp(input)
    exp = Transform(Exp(), torch.randn(128, 128))
    exp.run()

# @run
# def uniform():
#     class Uniform(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, a, b):
#             return torch.nn.init.uniform_(input, a, b, generator=None)
#     uniform = Transform(Uniform(), torch.randn(8, 128), 0, 1024.00)
#     uniform.run() #报错
    
# @run
def nll_loss_forward():
    class Nll_loss_forward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, target):
            return torch.nn.functional.nll_loss(input, target, weight=None, reduction="mean", ignore_index=-100)
    nll_loss_forward = Transform(Nll_loss_forward(), torch.randn(3, 5, requires_grad=True), torch.tensor([1, 0, 1]))
    nll_loss_forward.run()

# @run
# def nll_loss_backward():
#     class Null_loss_backward(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def backward(self, ) #参数设置问题
    
# @run
# def foreach_sqrt():
#     class Foreach_sqrt(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input):
#             return torch._foreach_sqrt(input)
#     foreach_sqrt = Transform(Foreach_sqrt(), (torch.randn(128, 128), torch.randn(128, 128)))
#     foreach_sqrt.run() #报错

# @run
def clamp():
    class Clamp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.clamp(input, min=-0.5, max=0.5)
    clamp = Transform(Clamp(), torch.randn(128, 128))
    clamp.run()

# @run
def ltScalar():
    class LtScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.lt(input, other)
    ltScalar = Transform(LtScalar(), torch.randn(128, 128), 0.5)
    ltScalar.run()

# @run
# def scaled_dot_product_attention():
#     class Scaled_dot_product_attention(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, query, key, value):
#             return torch.ops.aten.scaled_dot_product_attention(query, key, value)
#     scaled_dot_product_attention = Transform(Scaled_dot_product_attention(), torch.randn(32, 8, 128, 64), torch.randn(32, 8, 128, 64), torch.randn(32, 8, 128, 64))
#     scaled_dot_product_attention.run() #aten to linalg error

# @run
def reciprocal():
    class Reciprocal(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.reciprocal(input)
    reciprocal = Transform(Reciprocal(), torch.randn(128, 128))
    reciprocal.run()

# @run
def linalg_vector_norm():
    class Linalg_vector_norm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.linalg.vector_norm(input)
    linalg_vector_norm = Transform(Linalg_vector_norm(), torch.randn(128, 128))
    linalg_vector_norm.run()

# @run
#aten to linalg error
def native_dropout_backward():
    class Native_dropout_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, mask, scale):
            return torch.ops.aten.native_dropout_backward(grad_output, mask, scale)
    native_dropout_backward = Transform(Native_dropout_backward(), torch.randn(128, 128), torch.randn(128, 128), 3)
    native_dropout_backward.run()

# @run
def isnan():
    class Isnan(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.isnan(input)
    isnan = Transform(Isnan(), torch.randn(128, 128))
    isnan.run()

# @run
def cos():
    class Cos(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.cos(input)
    cos = Transform(Cos(), torch.randn(128, 128))
    cos.run()

# @run
def mm():
    class Mm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mat):
            return torch.mm(input, mat)
    mm = Transform(Mm(), torch.randn(8, 128), torch.randn(128, 8))
    mm.run()

# @run
#aten to linalg error
def index_put():
    class Index_put(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, indices, values, accumulate):
            return torch.ops.aten.index_put(input, indices, values, accumulate)
        
    index_put = Transform(Index_put(), torch.zeros(128, 128), (torch.Tensor([0, 1]), torch.Tensor([1, 2])), torch.randn([1, 1]), False)
    index_put.run()

# @run
def bmm():
    class Bmm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mat):
            return torch.bmm(input, mat)
        
    bmm = Transform(Bmm(), torch.randn(8, 128, 8), torch.randn(8, 8, 128))
    bmm.run()

# @run
def log():
    class Log(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.log(input)
    log = Transform(Log(), torch.randn(128, 128))
    log.run()

# @run
def mulScalar():
    class MulScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.mul(input, other)
        
    mulScalar = Transform(MulScalar(), torch.randn(128, 128), 3.14)
    mulScalar.run()

# @run
def addScalar():
    class AddScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.add(input, other)
    addScalar = Transform(AddScalar(), torch.randn(128, 128), 3.14)
    addScalar.run()

# @run
def isinf():
    class Isinf(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.isinf(input)
    isinf = Transform(Isinf(), torch.randn(128, 128))
    isinf.run()

# @run
def full_int():
    class Full_int(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, size, full_vale):
            return torch.full(size, full_vale)
    full_int = Transform(Full_int(), (128, 128), 2)
    full_int.run()

# @run
def softmax_backward():
    class Softmax_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, output, dim, input_dtype):
            return torch.ops.aten._softmax_backward_data(grad_output, output, dim, input_dtype)
    softmax_backward = Transform(Softmax_backward(), torch.randn(128, 128), torch.randn(128, 128), 1, 6)
    softmax_backward.run()

# @run
def expand():
    class Expand(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, size):
            return torch.ops.aten.expand(input, size)
    expand = Transform(Expand(), torch.randn(128, 128), (12, 128, 128))
    expand.run()

# @run
def batch_norm():
    class Batch_norm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, running_mean, running_var):
            return torch.nn.functional.batch_norm(input, running_mean, running_var)
    batch_norm = Transform(Batch_norm(), torch.randn(128, 128), torch.randn(128, 1), torch.randn(128, 1))
    batch_norm.run()

# @run
# def batch_norm_backward():
#     class Batch_norm_backward(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, )
    
# @run
#aten to linalg error
def embedding_dense_backward():
    class Embedding_dense_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
            return torch.ops.aten.embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
    embedding_dense_backward = Transform(Embedding_dense_backward(), torch.randn(128, 128), torch.randn(128, 128), 3200, -1, True)
    embedding_dense_backward.run()
        
# @run
# def embedding_bag():
#     class Embedding_bag(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, ) #参数设置问题
    
# @run
def eye():
    class Eye(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, n):
            return torch.eye(n)
    eye = Transform(Eye(), 128)
    eye.run()

# @run
#torch-mlir注册表中未实现该算子且 aten to linalg error
def frac():
    class Frac(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.frac(input)
    frac = Transform(Frac(), torch.randn(128, 128))
    frac.run()

# @run
# def grid_sampler_2d():
#     class Grid_sampler_2d(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, )

# @run
#aten to linalg error
def nantonum():
    class Nantonum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nan_to_num(input)
    nantonum = Transform(Nantonum(), torch.randn(128, 128))
    nantonum.run()

# @run
#aten to linalg error
def linspace():
    class Linspace(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, start, end, step):
            return torch.linspace(start, end, step)
    linspace = Transform(Linspace(), 0, 30, 5)
    linspace.run()

# @run
def log10():
    class Log10(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.log10(input)
    log10 = Transform(Log10(), torch.randn(128, 128))
    log10.run()

# @run
def log1p():
    class Log1p(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.log1p(input)
    log1p = Transform(Log1p(), torch.randn(128, 128))
    log1p.run()

# @run
#aten to linalg error
def xlogy():
    class Xlogy(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.xlogy(input, other)
    xlogy = Transform(Xlogy(), torch.randn(128, 128), torch.randn(128, 128))
    xlogy.run()

# @run
#aten to linalg error
def logspace():
    class Logspace(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, start, end, steps):
            return torch.logspace(start, end, steps, base=10.0)
    log1p = Transform(Logspace(), 0, 128, 1)
    log1p.run()

# @run
def log_softmax():
    class Log_softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.nn.functional.log_softmax(input, dim)
    log_softmax = Transform(Log_softmax(), torch.randn(128, 128), 0)
    log_softmax.run()

# @run
def log_softmax_backward_data():
    class Log_softmax_backward_data(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, output, dim, input_dtype):
            return torch.ops.aten._log_softmax_backward_data(grad_output, output, dim, input_dtype)
    log_softmax_backward_data = Transform(Log_softmax_backward_data(), torch.randn(128, 128), torch.randn(128, 128), 0, 6)
    log_softmax_backward_data.run()

# @run
def amin():
    class Amin(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim):
            return torch.amin(input, dim, keepdim=False)
    amin = Transform(Amin(), torch.randn(128, 128), 0)
    amin.run()

# @run
#参数设置问题
def max_pool2d():
    class Max_pool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, kernel):
            return torch.nn.functional.max_pool2d(input, kernel)
    max_pool2d = Transform(Max_pool2d(), torch.randn(128, 128), (1,))
    max_pool2d.run()

# @run
#torch-mlir注册表未实现该算子
def median():
    class Median(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.median(input)
    median = Transform(Median(), torch.randn(128, 128))
    median.run()

# @run
def repeat():
    class Repeat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, repeats):
            return torch.repeat_interleave(input, repeats)
    repeat = Transform(Repeat(), torch.randn(128, 128), 2)
    repeat.run()

# @run
def round():
    class Round(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.round(input)
    round = Transform(Round(), torch.randn(128, 128))
    round.run()

# @run
def prelu():
    class Prelu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, weight):
            return torch.nn.functional.prelu(input, weight)
    prelu = Transform(Prelu(), torch.randn(128, 128), torch.ones(128,))
    prelu.run()

# @run
def gelu():
    class Gelu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.gelu(input, approximate='none')
    gelu = Transform(Gelu(), torch.randn(128, 128))
    gelu.run()

# @run
def gelu_backward():
    class Gelu_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, output):
            return torch.ops.aten.gelu_backward(grad_output, output, approximate='none')
    gelu_backward = Transform(Gelu_backward(), torch.randn(128, 128), torch.randn(128, 128))
    gelu_backward.run()

# @run
def mish():
    class Mish(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.mish(input)
    mish = Transform(Mish(), torch.randn(128, 128))
    mish.run()

# @run 
def sin():
    class Sin(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.sin(input)
    sin = Transform(Sin(), torch.randn(128, 128))
    sin.run()

# @run
def sinh():
    class Sinh(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.sinh(input)
    sinh = Transform(Sinh(), torch.randn(128, 128))
    sinh.run()

# @run
def tan():
    class Tan(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.tan(input)
    tan = Transform(Tan(), torch.randn(128, 128))
    tan.run()

# @run
def sumdim_Intlist():
    class Sumdim_Intlist(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input, dim):
            return torch.sum(input, dim, keepdim=False)
    sumdim_Intlist = Transform(Sumdim_Intlist(), torch.randn(128, 128), (1, ))
    sumdim_Intlist.run()

# @run
#torch-mlir注册表中没有该算子
def nansum():
    class Nansum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nansum(input)
    nansum = Transform(Nansum(), torch.randn(128, 128))
    nansum.run()

# @run
#torch-mlir注册表中没有该算子
def proddim_int():
    class Proddim_int(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.prod(input)
    proddim_int = Transform(Proddim_int(), torch.randn(128, 128))
    proddim_int.run()

# @run
def threshold():
    class Threshold(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, threshold, value):
            return torch.nn.functional.threshold(input, threshold, value)
    threshold = Transform(Threshold(), torch.randn(128, 128), 0.1, 1.0)
    threshold.run()

# @run
def threshold_backward():
    class Threshold_backward(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grad_output, slef, threshold):
            return torch.ops.aten.threshold_backward(grad_output, slef, threshold)
    threshold_backward = Transform(Threshold_backward(), torch.randn(128, 128), torch.randn(128, 128), 0.1)
    threshold_backward.run()

# @run
def flip():
    class Flip(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dims):
            return torch.flip(input, dims)
    flip = Transform(Flip(), torch.randn(128, 128), (1,))
    flip.run()

# @run
def roll():
    class Roll(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, shifts, dims):
            return torch.roll(input, shifts, dims)
    roll = Transform(Roll(), torch.randn(128, 128), (2, 1), (0, 1))
    roll.run()

# @run
#torch-mlir注册表中未实现
def trunc():
    class Trunc(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.trunc(input)
    trunc = Transform(Trunc(), torch.randn(128, 128))
    trunc.run()

# @run
#参数设置问题
def unique_consecutive():
    class Unique_consecutive(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.unique_consecutive(input, return_inverse=False, return_counts=False, dim=0)
    unique_consecutive = Transform(Unique_consecutive(), torch.tensor([1, 1, 2, 2, 3, 1, 1, 2]))
    unique_consecutive.run()

# @run
def var():
    class Var(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.var(input, dim=None, correction=1, keepdim=False, out=None)
    var = Transform(Var(), torch.randn(128, 128))
    var.run()

# @run
def addmm():
    class Addmm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mat1, mat2):
            return torch.addmm(input, mat1, mat2)
    addmm = Transform(Addmm(), torch.randn(128, 128), torch.randn(128, 256), torch.randn(256, 128))
    addmm.run()

# @run
def masked_fill_Scalar():
    class Masked_fill_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mask, value):
            return torch.ops.aten.masked_fill_.Scalar(input, mask, value)
    masked_fill_Scalar = Transform(Masked_fill_Scalar(), torch.randn(128, 128), torch.ones((128, 128), dtype=torch.bool), 2)
    masked_fill_Scalar.run()

# @run
def masked_fill_Tensor():
    class Masked_fill_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mask, value):
            return torch.ops.aten.masked_fill_.Tensor(input, mask, value)
    masked_fill_Tensor = Transform(Masked_fill_Tensor(), torch.randn(128, 128), torch.ones((128, 128), dtype=torch.bool), torch.tensor(5))
    masked_fill_Tensor.run()

# @run
#aten to linalg error
def masked_scatter():
    class Masked_scatter(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, mask, source):
            return torch.ops.aten.masked_scatter(input, mask, source)
    masked_scatter = Transform(Masked_scatter(), torch.randn(128, 128), torch.ones((128, 128), dtype=torch.bool), torch.randn(128, 128))
    masked_scatter.run()

# @run
def scatter_src():
    class Scatter_src(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index, src):
            return torch.ops.aten.scatter.src(input, dim, index, src)
    scatter_src = Transform(Scatter_src(), torch.randn(128, 128), 1, torch.ones((128, 128), dtype=torch.int64), torch.randn(128, 128))
    scatter_src.run()

# @run
def scatter_value():
    class Scatter_value(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index, src):
            return torch.ops.aten.scatter.value(input, dim, index, src)
    scatter_value = Transform(Scatter_value(), torch.randn(128, 128), 1, torch.ones((128, 128), dtype=torch.int64), 5)
    scatter_value.run()

# @run
#aten to linalg error
def scatter_reduce():
    class Scatter_reduce(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index, src):
            return torch.ops.aten.scatter.reduce(input, dim, index, src, reduce='add')
    scatter_reduce = Transform(Scatter_reduce(), torch.randn(128, 128), 1, torch.ones((128, 128), dtype=torch.int64), torch.randn(128, 128))
    scatter_reduce.run()

# @run
#aten to linalg error
def scatter_add():
    class Scatter_add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, dim, index, src):
            return torch.scatter_add(input, dim, index, src)
    scatter_add = Transform(Scatter_add(), torch.randn(128, 128), 1, torch.ones((128, 128), dtype=torch.int64), torch.randn(128, 128))
    scatter_add.run()

# @run
def bitwise_and():
    class Bitwise_and(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.bitwise_and(input, other)
    bitwise_and = Transform(Bitwise_and(), torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    bitwise_and.run()

# @run
def bitwise_or():
    class Bitwise_or(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.bitwise_or(input, other)
    bitwise_or = Transform(Bitwise_or(), torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    bitwise_or.run()
    
# @run
def bitwise_xor():
    class Bitwise_xor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.bitwise_xor(input, other)
    bitwise_xor = Transform(Bitwise_xor(), torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    bitwise_xor.run()
    
# @run
def bitwise_left_shift():
    class Bitwise_left_shift(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.bitwise_left_shift(input, other)
    bitwise_left_shift = Transform(Bitwise_left_shift(), torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    bitwise_left_shift.run()
    
# @run
def bitwise_right_shift():
    class Bitwise_right_shift(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.bitwise_right_shift(input, other)
    bitwise_right_shift = Transform(Bitwise_right_shift(), torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    bitwise_right_shift.run()

# @run
#aten to linalg error
def addbmm():
    class Addbmm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, batch1, batch2):
            return torch.addbmm(input, batch1, batch2, out=torch.randn(8, 128, 128))
    addbmm = Transform(Addbmm(), torch.randn(128, 128), torch.randn(8, 128, 8), torch.randn(8, 8, 128))
    addbmm.run()

# @run
# def random_from():
#     class Random_from(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, form_, to_, generator):
#             return torch.ops.aten.random.from(Random_from(), input, from_, to_, generator)
#     random_from = Transform(Random_from(), )

# @run
# def exponential():
#     class Exponential(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, input, lambd, generator):
#             return torch.ops

# @run
#aten to linalg error
def triu():
    class Triu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.triu(input, diagonal=0)
    triu = Transform(Triu(), torch.randn(128, 128))
    triu.run()

# @run
def tril_indices():
    class Tril_indices(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, row, col):
            return torch.tril_indices(row, col)
    tril_indices = Transform(Tril_indices(), 128, 128)
    tril_indices.run()

# @run
def trace():
    class Trace(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.trace(input)
    trace = Transform(Trace(), torch.randn(128, 128))
    trace.run()

# @run
def ne_Tensor():
    class Ne_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.ne(input, other)
    ne_Tensor = Transform(Ne_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    ne_Tensor.run()

# @run
def eq_Tensor():
    class Eq_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.eq(input, other)
    eq_Tensor = Transform(Eq_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    eq_Tensor.run()

# @run
def eq_Scalar():
    class Eq_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.eq(input, other)
    eq_Scalar = Transform(Eq_Scalar(), torch.randn(128, 128), 1.0)
    eq_Scalar.run()

# @run
def ge_Scalar():
    class Ge_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.ge(input, other)
    ge_Scalar = Transform(Ge_Scalar(), torch.randn(128, 128), 1.0)
    ge_Scalar.run()

# @run
def ge_Tensor():
    class Ge_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.ge(input, other)
    ge_Tensor = Transform(Ge_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    ge_Tensor.run()

# @run
def le_Scalar():
    class Le_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.le(input, other)
    le_Scalar = Transform(Le_Scalar(), torch.randn(128, 128), 1.0)
    le_Scalar.run()

# @run
def le_Tensor():
    class Le_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.le(input, other)
    le_Tensor = Transform(Le_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    le_Tensor.run()

# @run
def gt_Scalar():
    class Gt_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.gt(input, other)
    gt_Scalar = Transform(Gt_Scalar(), torch.randn(128, 128), 1.0)
    gt_Scalar.run()

# @run
def gt_Tensor():
    class Gt_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.gt(input, other)
    gt_Tensor = Transform(Gt_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    gt_Tensor.run()

# @run
def lt_Scalar():
    class Lt_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.lt(input, other)
    lt_Scalar = Transform(Lt_Scalar(), torch.randn(128, 128), 1.0)
    lt_Scalar.run()

# @run
def lt_Tensor():
    class Lt_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.lt(input, other)
    lt_Tensor = Transform(Lt_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    lt_Tensor.run()

# @run
#aten to linalg error
def nonzero_out():
    class Nonzero_out(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nonzero(input)
    nonzero_out = Transform(Nonzero_out(), torch.randn(128, 128))
    nonzero_out.run()

# @run
def addcmul():
    class Addcmul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, tensor1, tensor2):
            return torch.addcmul(input, tensor1, tensor2)
    addcmul = Transform(Addcmul(), torch.randn(128, 128), torch.randn(128, 1), torch.randn(1, 128))
    addcmul.run()

# @run
def addcdiv():
    class Addcdiv(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, tensor1, tensor2):
            return torch.addcdiv(input, tensor1, tensor2)
    addcdiv = Transform(Addcdiv(), torch.randn(128, 128), torch.randn(128, 1), torch.randn(1, 128))
    addcdiv.run()

# @run
#aten to linalg error
def multinomial():
    class Multinomial(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, num_samples):
            return torch.multinomial(input, num_samples)
    weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
    multinomial = Transform(Multinomial(), weights, 2)
    multinomial.run()

# @run
#aten to linalg error
def erfinv():
    class Erfinv(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.erfinv(input)
    erfinv = Transform(Erfinv(), torch.randn(128, 128))
    erfinv.run()

# @run
def sign():
    class Sign(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.sign(input)
    sign = Transform(Sign(), torch.randn(128, 128))
    sign.run()

# @run
def signbit():
    class Signbit(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.signbit(input)
    signbit = Transform(Signbit(), torch.randn(128, 128))
    signbit.run()

# @run
def lerp_Tensor():
    class Lerp_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, end, weight):
            return torch.lerp(input, end, weight)
    lerp_Tensor = Transform(Lerp_Tensor(), torch.randn(128, 128), torch.randn(128, 128), torch.randn(128, 128))
    lerp_Tensor.run()

# @run
def lerp_Scalar():
    class Lerp_Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, end, weight):
            return torch.lerp(input, end, weight)
    lerp_Scalar = Transform(Lerp_Scalar(), torch.randn(128, 128), torch.randn(128, 128), 0.5)
    lerp_Scalar.run()

# @run
#注册表中没有该算子
def histc():
    class Histc(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.histc(input)
    histc = Transform(Histc(), torch.randn(128, 128))
    histc.run()

# @run
def fmod():
    class Fmod(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.fmod(input, other)
    fmod = Transform(Fmod(), torch.randn(128, 128), torch.randn(128, 128))
    fmod.run()

# @run
#torch-mlir注册表中未实现该算子
def hypot():
    class Hypot(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.hypot(input, other)
    hypot = Transform(Hypot(), torch.randn(128, 128), torch.randn(128, 128))
    hypot.run()

# @run
def remainder():
    class Remainder(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, a, b):
            return torch.remainder(a, b)
    remainder = Transform(Remainder(), torch.randn(128, 128), torch.randn(128, 128))
    remainder.run()

# @run
def minimum():
    class Minimum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.minimum(a, b)
    minimum = Transform(Minimum(), torch.randn(128, 128), torch.randn(128, 128))
    minimum.run()

# @run
def sort():
    class Sort(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sort(a)
    sort = Transform(Sort(), torch.randn(128, 128))
    sort.run()

# @run
def argsort():
    class Argsort(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.argsort(input)
    argsort = Transform(Argsort(), torch.randn(128, 128))
    argsort.run()

# @run
def topk():
    class Topk(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, k):
            return torch.topk(a, k)
    topk = Transform(Topk(), torch.randn(128, 128), 5)
    topk.run()

# @run
#aten to linalg error
def all():
    class All(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.all(a)
    all = Transform(All(), torch.randn(1, 2).bool())
    all.run()

# @run
#aten to linalg error
def any():
    class Any(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.any(a)
    any = Transform(Any(), torch.randn(1, 2).bool())
    any.run()

# @run
def unfold():
    class Unfold(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, kernel_size):
            return torch.nn.functional.unfold(input, kernel_size)
    unfold = Transform(Unfold(), torch.randn(2, 5, 3, 4), (2,3))
    unfold.run()

# @run
#注册表中未实现
def equal():
    class Equal(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, other):
            return torch.equal(input, other)
    equal = Transform(Equal(), torch.randn(128, 128), torch.randn(128, 128))
    equal.run()

# @run
def powTensor_Tensor():
    class PowTensor_Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, exponent):
            return torch.pow(input, exponent)
    powTensor_Tensor = Transform(PowTensor_Tensor(), torch.randn(128, 128), torch.randn(128, 128))
    powTensor_Tensor.run()

# @run
def powScalar():
    class PowScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, exponent):
            return torch.pow(input, exponent)
    powScalar = Transform(PowScalar(), 5, torch.randn(128, 128))
    powScalar.run()

# @run
def normal_():
    class Normal_(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.Tensor.normal_(input)
    normal_ = Transform(Normal_(), torch.empty(128, 128))
    normal_.run()

# @run
def normalTensorfloat():
    class NormalTensorfloat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, mean, std):
            return torch.Tensor.normal_(mean, std)
    normalTensorfloat = Transform(NormalTensorfloat(), torch.randn(128, 128), 5)
    normalTensorfloat.run()

@run
def normalfloatTensor():
    class NormalfloatTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, mean, std):
            return torch.normal(mean, std)
    normalfloatTensor = Transform(NormalfloatTensor(), 0.0, torch.randn(128, 128))
    normalfloatTensor.run()







<<<<<<< HEAD
@run
def test_dropout():
    class Dropout(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x:float) -> float:
            return torch.nn.Dropout(x)
    dropout = Transform(Dropout(), 0.3)
    dropout.run()
=======
>>>>>>> add Cumsum elu glu logical_not transpose smooth_l1_loss
=======
>>>>>>> all kernel aten to linalg
