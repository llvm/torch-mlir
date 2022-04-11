import torch
from functorch.compile import memory_efficient_fusion, get_decompositions, default_partition
from torch import fx
import copy


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


class AOTModule:

    def __init__(self, model, inputs, labels, training_fn):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.training_fn = training_fn
        self.forward_graph = None
        self.backward_graph = None
        self.forward_inputs = None
        self.backward_inputs = None

    def change_fx_graph_return_to_tuple(self, fx_g: fx.GraphModule):
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
                    # a tuple for it.
                    if len(out_nodes) == 1:
                        node.args = out_nodes
                    else:
                        node.args = (tuple(out_nodes), )
        fx_g.graph.lint()
        fx_g.recompile()
        return fx_g

    def get_forward_graph(self, fx_g: fx.GraphModule, inps):
        return_fx = copy.deepcopy(fx_g)
        f = self.change_fx_graph_return_to_tuple(fx_g)
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        torch.jit.save(f, "forw.pt")
        f = torch.jit.load("forw.pt")
        self.forward_graph = f
        self.forward_inputs = copy.deepcopy(inps)
        return return_fx

    def get_backward_graph(self, fx_g: fx.GraphModule, inps):
        return_fx = copy.deepcopy(fx_g)
        f = self.change_fx_graph_return_to_tuple(fx_g)
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        torch.jit.save(f, "back.pt")
        f = torch.jit.load("back.pt")
        self.backward_graph = f
        self.backward_inputs = copy.deepcopy(inps)
        return return_fx

    def generate_graphs(self):
        aot_model = memory_efficient_fusion(
            self.model,
            fw_compiler=self.get_forward_graph,
            bw_compiler=self.get_backward_graph,
            partition_fn=default_partition,
            decompositions=get_decompositions([
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward
            ]))
        self.training_fn(aot_model, self.inputs, self.labels)
