# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch._decomp import get_decompositions
from functorch import make_fx
from torch.nn.utils import _stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch import fx
import copy
import tempfile

torch.manual_seed(0)

############################## Utility Functions ##############################


def get_input_annotations(inputs: tuple, dynamic: bool) -> list:
    """Generates the annotation i.e., shape and dtype for the given inputs, required by torch-mlir module."""

    annotations_list = [None]
    for i in inputs:
        temp_list = []
        if dynamic:
            temp_list.append([-1 for i in range(len(i.shape))])
        else:
            temp_list.append(list(i.shape))
        temp_list.append(i.dtype)
        temp_list.append(True)
        annotations_list.append(tuple(temp_list))
    return annotations_list


def change_fx_graph_return_to_tuple(fx_g: fx.GraphModule):
    for node in fx_g.graph.nodes:
        if node.op == "output":
            # output nodes always have one argument
            node_arg = node.args[0]
            out_nodes = []
            if isinstance(node_arg, list):
                # Don't return NoneType elements.
                for out_node in node_arg:
                    if not isinstance(out_node, type(None)):
                        out_nodes.append(out_node)
                # If there is a single tensor/element to be returned don't
                # create a tuple for it.
                if len(out_nodes) == 1:
                    node.args = out_nodes
                else:
                    node.args = (tuple(out_nodes), )
    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def generate_graph(model, inputs, training_fn):
    # TODO: Pass the decomposition_table according to the model/needs.
    fx_g = make_fx(training_fn,
                   decomposition_table=get_decompositions([
                       torch.ops.aten.embedding_dense_backward,
                       torch.ops.aten.native_layer_norm_backward,
                       torch.ops.aten.slice_backward,
                       torch.ops.aten.select_backward
                   ]))(dict(model.named_parameters()),
                       dict(model.named_buffers()), inputs)
    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()
    fx_g = change_fx_graph_return_to_tuple(fx_g)
    ts_g = torch.jit.script(fx_g)
    # TODO: If not saved/load creates some unnecessary functions that
    # causes problem during mlir-translate.
    temp=tempfile.NamedTemporaryFile(suffix='_heavy_dep',
                                 prefix='temp_ts_')
    ts_g.save(temp.name)
    new_ts = torch.jit.load(temp.name)
    return new_ts


def getAnnotatedModule(ts_module, inputs):
    export(ts_module.forward)
    annotate_args_decorator = annotate_args(
        get_input_annotations(inputs, dynamic=False))
    annotate_args_decorator(ts_module.forward)
    return ts_module

    ############################################################################


# Basic NeuralNet training test.
# This trains the Neural Net and returns the updated weights with the
# sgd optimzier.


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


neural_net_model = NeuralNet()
input = torch.randn(1, 10)


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]


# TODO: Pass and update the optimizer fn. Currently, we don't support passing
# elemental types.
def training_fn(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(neural_net_model, params_and_buffers, args,
                               {}).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    optim.step()
    return params, buffers


# We need to pass the model parameters, buffers and the inputs respectively in
# order.
training_inputs = [i.detach() for i in neural_net_model.parameters()]
for i in neural_net_model.buffers():
    training_inputs.append(i.detach())

training_inputs.append(input)

neural_net_ts = generate_graph(neural_net_model, (input, ), training_fn)

neural_net_ts_annotated = getAnnotatedModule(neural_net_ts, training_inputs)


@register_test_case(module_factory=lambda: neural_net_ts_annotated)
def NeuralNet_training_basic(module, tu: TestUtils):
    module.forward(*training_inputs)


##############################################################################

# Bert training.


class MiniLMSequenceClassification(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            # The below model is a variant of
            # `microsoft/MiniLM-L12-H384-uncased` with less parameters.
            "nreimers/MiniLM-L6-H384-uncased",  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


bert_model = MiniLMSequenceClassification()
input = torch.randint(2, (1, 128))


# TODO: Pass and update the optimizer fn. Currently, we don't support passing
# elemental types.
def training_fn(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(bert_model, params_and_buffers, args,
                               {}).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    optim.step()
    return params, buffers


# We need to pass the model parameters, buffers and the inputs respectively in
# order.
bert_inputs = [i.detach() for i in bert_model.parameters()]
for i in bert_model.buffers():
    bert_inputs.append(i.detach())

bert_inputs.append(input)

bert_ts = generate_graph(bert_model, (input, ), training_fn)

bert_ts_annotated = getAnnotatedModule(bert_ts, bert_inputs)


@register_test_case(module_factory=lambda: bert_ts_annotated)
def BERT_training_basic(module, tu: TestUtils):
    module.forward(*bert_inputs)
