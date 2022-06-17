# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from transformers import AutoTokenizer, OPTConfig, OPTModel, GPT2Tokenizer

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

torch.manual_seed(0)


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return torch.tensor([tokenizer.encode(sentence)])


def getTracedRecursiveScriptModule(module, trace_input):
    traced_module = torch.jit.trace_module(module, trace_input)
    script_module = traced_module._actual_script_module
    export(script_module.forward)
    annotate_args_decorator = annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    annotate_args_decorator(script_module.forward)
    return script_module


class HfOPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        configuration = OPTConfig()
        configuration.return_dict = False
        self.model = OPTModel(configuration)
        self.model.eval()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, input_ids, attention_mask):
        return self.model.forward(input_ids, attention_mask)


# ==============================================================================

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
trace_input = {
    "forward":
        (inputs.data["input_ids"], inputs.data["attention_mask"])
}


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    HfOPT(), trace_input))
def HfOPT_basic(module, tu: TestUtils):
    module.forward(trace_input)
