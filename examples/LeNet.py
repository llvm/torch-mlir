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
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=(2, 2))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet5()
print(net)

# NCHW layout in pytorch
module = torch_mlir.compile(net, torch.ones(1, 1, 32, 32), output_type="torch")
print(module.operation.get_asm(large_elements_limit=10))
# module_linalg = torch_mlir.compile(
#     net, torch.ones(1, 1, 32, 32), output_type="linalg-on-tensors"
# )
# print(module_linalg.operation.get_asm(large_elements_limit=10))

torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module, "builtin.module(torch-obfuscate-ops-pipeline)", "obfuscate torch IR"
)
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module,
    "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
    "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
)
print(module.operation.get_asm(large_elements_limit=10))


backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)
jit_func = jit_module.forward
print(jit_func(torch.ones(1, 1, 32, 32).numpy()))
print(jit_func(torch.zeros(1, 1, 32, 32).numpy()))
