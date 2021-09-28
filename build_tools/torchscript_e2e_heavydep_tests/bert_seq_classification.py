# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Basic BertForSequenceClassification program to classify the input sentence.

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

torch.manual_seed(0)
CLS = "[CLS]"
SEP = "[SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def _prepare_sentence_tokens(sentence: str):
    return torch.tensor([tokenizer.encode(sentence)])


class BasicBertSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True)
        self.model.eval()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, tokens):
        return self.model.forward(tokens)[0]


trace_input = {
    'forward': _prepare_sentence_tokens("how do you like the project")
}

test_input = _prepare_sentence_tokens("this project is very interesting")


def getTracedRecursiveScriptModule():
    traced_module = torch.jit.trace_module(BasicBertSequenceClassification(),
                                           trace_input)
    script_module = traced_module._actual_script_module
    export(script_module.forward)
    annotate_args_decorator = annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    annotate_args_decorator(script_module.forward)
    return script_module


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule())
def BasicBertSequenceClassification_basic(module, tu: TestUtils):
    module.forward(test_input)
