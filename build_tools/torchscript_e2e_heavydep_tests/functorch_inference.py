# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BartConfig, BigBirdConfig, GPT2Config, DistilBertConfig
import transformers
import torch
import torch.utils._pytree as pytree

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from functorch_utils import AOTModule, get_input_annotations

torch.manual_seed(0)


def getAnnotatedModule(ts_module, inputs):
    export(ts_module.forward)
    annotate_args_decorator = annotate_args(
        get_input_annotations(inputs, dynamic=False))
    annotate_args_decorator(ts_module.forward)
    return ts_module


pytree._register_pytree_node(
    transformers.modeling_outputs.Seq2SeqLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.Seq2SeqLMOutput(
        loss=values[1], logits=values[1]),
)

pytree._register_pytree_node(
    transformers.modeling_outputs.MaskedLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.MaskedLMOutput(
        loss=values[1], logits=values[1]),
)

pytree._register_pytree_node(
    transformers.modeling_outputs.CausalLMOutputWithCrossAttentions,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.
    CausalLMOutputWithCrossAttentions(loss=values[1], logits=values[1]),
)

torch.manual_seed(42)

# ==============================================================================
# BART Model

config = BartConfig()
model_type = AutoModelForSeq2SeqLM
input_size = (1, 2)
device = "cpu"
dtype = torch.float

model = model_type.from_config(config).to(device, dtype=dtype)
input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
train_inputs = {"input_ids": input_ids, "labels": decoder_ids}


def inference_fn(model, input, labels):
    return model(**input).loss.sum().backward()


aot_module = AOTModule(model,
                       train_inputs,
                       labels=None,
                       training_fn=inference_fn)
aot_module.generate_graphs()

forw_inputs = []
for inp in aot_module.forward_inputs:
    forw_inputs.append(inp.detach())

ts_module_forw = getAnnotatedModule(aot_module.forward_graph,
                                    aot_module.forward_inputs)


@register_test_case(module_factory=lambda: ts_module_forw)
def BartSequenceClassification_basic(module, tu: TestUtils):
    module.forward(*forw_inputs)


# ==============================================================================
# BigBird Model

config = BigBirdConfig(attention_type="original_full")
model_type = AutoModelForMaskedLM
input_size = (1, 2)
device = "cpu"
dtype = torch.float

model = model_type.from_config(config).to(device, dtype=dtype)
input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
train_inputs = {"input_ids": input_ids, "labels": decoder_ids}


def inference_fn(model, input, labels):
    return model(**input).loss.sum().backward()


aot_module = AOTModule(model,
                       train_inputs,
                       labels=None,
                       training_fn=inference_fn)
aot_module.generate_graphs()

forw_inputs = []
for inp in aot_module.forward_inputs:
    forw_inputs.append(inp.detach())

ts_module_forw = getAnnotatedModule(aot_module.forward_graph,
                                    aot_module.forward_inputs)


@register_test_case(module_factory=lambda: ts_module_forw)
def BigBirdSequenceClassification_basic(module, tu: TestUtils):
    module.forward(*forw_inputs)


# ==============================================================================
# GPT2 Model

config = GPT2Config()
model_type = AutoModelForCausalLM
input_size = (1, 2)
device = "cpu"
dtype = torch.float

model = model_type.from_config(config).to(device, dtype=dtype)
input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
train_inputs = {"input_ids": input_ids, "labels": decoder_ids}


def inference_fn(model, input, labels):
    return model(**input).loss.sum().backward()


aot_module = AOTModule(model,
                       train_inputs,
                       labels=None,
                       training_fn=inference_fn)
aot_module.generate_graphs()

forw_inputs = []
for inp in aot_module.forward_inputs:
    forw_inputs.append(inp.detach())

ts_module_forw = getAnnotatedModule(aot_module.forward_graph,
                                    aot_module.forward_inputs)


@register_test_case(module_factory=lambda: ts_module_forw)
def GPT2SequenceClassification_basic(module, tu: TestUtils):
    module.forward(*forw_inputs)
