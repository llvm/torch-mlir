# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# A pretrained model to classify the input sentence.
# https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    annotate_args_decorator = annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    annotate_args_decorator(script_module.forward)
    return script_module


class HfSequenceClassification(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, tokens):
        return self.model.forward(tokens)[0]


# ==============================================================================

hf_minilm_model = "philschmid/MiniLM-L6-H384-uncased-sst2"

trace_input = {
    "forward": prepare_sentence_tokens(hf_minilm_model, "how do you like the project")
}
test_input = prepare_sentence_tokens(
    hf_minilm_model, "this project is very interesting")


@register_test_case(
    module_factory=lambda: getTracedRecursiveScriptModule(
        HfSequenceClassification(hf_minilm_model), trace_input
    )
)
def MiniLMSequenceClassification_basic(module, tu: TestUtils):
    module.forward(test_input)

# ==============================================================================

hf_albert_model = "albert-base-v2"

trace_input = {
    "forward": prepare_sentence_tokens(hf_albert_model, "how do you like the project")
}
test_input = prepare_sentence_tokens(
    hf_albert_model, "this project is very interesting")


@register_test_case(
    module_factory=lambda: getTracedRecursiveScriptModule(
        HfSequenceClassification(hf_albert_model), trace_input
    )
)
def AlbertSequenceClassification_basic(module, tu: TestUtils):
    module.forward(test_input)

# ==============================================================================
