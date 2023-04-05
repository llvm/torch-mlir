import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input shape is 1x1x28x28
        # Max pooling over a (2, 2) window, if use default stride, error will happen
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=(2, 2))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet5()
net.eval()
print(net)


module = torch_mlir.compile(net, torch.ones(1, 1, 28, 28), output_type="torch")
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module,
    "builtin.module(func.func(torch-branch-layer{layer=2 branch=3}))",
    "BranchLayer",
)
#print(module.operation.get_asm(large_elements_limit=10))
#'''
print("================")
print("after lower to linalg")
print("================")
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module,
    "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
    "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
)
#print(module.operation.get_asm(large_elements_limit=10))

print("================")
print("run model")
print("================")
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)
jit_func = jit_module.forward
out1 = jit_func(torch.ones(1, 1, 28, 28).numpy())
out2 = jit_func(torch.zeros(1, 1, 28, 28).numpy())
print("output:")
print(out1)
print(out2)

module_origin = torch_mlir.compile(
    net, torch.ones(1, 1, 28, 28), output_type="linalg-on-tensors"
)
jit_func_origin = backend.load(backend.compile(module_origin)).forward
out1_origin = jit_func_origin(torch.ones(1, 1, 28, 28).numpy())
out2_origin = jit_func_origin(torch.zeros(1, 1, 28, 28).numpy())
print("origin output:")
print(out1_origin)
print(out2_origin)

print("diffs:")
print(out1 - out1_origin)
print(out2 - out2_origin)
#'''


