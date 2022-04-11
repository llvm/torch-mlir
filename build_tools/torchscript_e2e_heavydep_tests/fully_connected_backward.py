# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from functorch_utils import AOTModule, get_input_annotations

torch.manual_seed(0)

def getAnnotatedModule(ts_module, inputs):
  export(ts_module.forward)
  annotate_args_decorator = annotate_args(
      get_input_annotations(inputs, dynamic=True))
  annotate_args_decorator(ts_module.forward)
  return ts_module

class NeuralNet(torch.nn.Module):

  def __init__(self):
    super(NeuralNet, self).__init__()
    self.l1 = torch.nn.Linear(10, 16)
    self.relu = torch.nn.ReLU()
    self.l2 = torch.nn.Linear(16, 2)

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out


input = torch.randn(1, 10)
labels = torch.randn(1, 2)


def run_model(module, input, labels):
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
  iters = 1
  for _ in range(iters):
    optimizer.zero_grad()
    output = module(input)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()


aot_module = AOTModule(NeuralNet(), input, labels, run_model)
aot_module.generate_graphs()

# ==============================================================================
# Forward test.
forw_inputs = []
for inp in aot_module.forward_inputs:
  forw_inputs.append(inp.detach())

ts_module_forw = getAnnotatedModule(aot_module.forward_graph,
                                    aot_module.forward_inputs)


@register_test_case(module_factory=lambda: ts_module_forw)
def NeuralNet_forward_basic(module, tu: TestUtils):
  module.forward(*forw_inputs)


# ==============================================================================
# Backward test.
back_inputs = []
for inp in aot_module.backward_inputs:
  back_inputs.append(inp.detach())

ts_module_back = getAnnotatedModule(aot_module.backward_graph,
                                    aot_module.backward_inputs)

@register_test_case(module_factory=lambda: ts_module_back)
def NeuralNet_backward_basic(module, tu: TestUtils):
  module.forward(*back_inputs)
