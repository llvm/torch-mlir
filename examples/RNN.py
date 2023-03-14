import torch
from torch import nn
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class RNN_scratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        torch.manual_seed(2)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = torch.zeros(1, self.hidden_size)
        self.train(False)

    def forward(self, input):
        hidden = self.hidden
        for i in range(input.size(0)):
            combined = torch.cat((input[i], hidden), 1)
            hidden = torch.sigmoid(self.i2h(combined))
            output = self.i2o(combined)
        return output, hidden


sequence_len = 3
batch_num = 1
input_size, hidden_size, output_size = 10, 20, 18

data_in = torch.zeros((sequence_len, batch_num, input_size))
rnn = RNN_scratch(input_size, hidden_size, output_size)

module = torch_mlir.compile(
    rnn, data_in, output_type="torch", use_tracing=True, ignore_traced_shapes=True
)
print(module.operation.get_asm(large_elements_limit=10))
