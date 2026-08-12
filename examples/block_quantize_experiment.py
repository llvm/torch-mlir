import torch
import torch.nn as nn

import qtorch
from qtorch.quant import block_quantize

import torch_mlir

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_size):
        super(SimpleModel, self).__init__()
        self.matmul = nn.Linear(input_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        matmul_out = self.matmul(x.flatten(1))
        quantized_matmul_out = block_quantize(matmul_out, wl=8, dim=0, rounding="nearest")
        relu_out = self.relu(quantized_matmul_out)
        return relu_out

batches = 5
input_dim = 64
output_size = 4
inputs = torch.randn(batches, input_dim)
model = SimpleModel(input_dim, output_size)
print("forward propagate results on inputs is:\n", model.forward(inputs))

# quantized_inputs = block_quantize(inputs, wl=8, dim=0, rounding="nearest")
# print("forward propagate of quantized inputs result is ", model.forward(quantized_inputs))

module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.TOSA)
print("Module compiled to TOSA is:\n", module)
