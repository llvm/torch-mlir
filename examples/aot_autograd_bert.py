import torch
import torch.utils._pytree as pytree

from functorch.compile import aot_module, aot_function
from functorch.compile import nop
from functorch.compile import get_decompositions

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import transformers
from transformers import BertForMaskedLM

pytree._register_pytree_node(transformers.modeling_outputs.MaskedLMOutput, lambda x: (
    [x.logits], None), lambda values, _: transformers.modeling_outputs.MaskedLMOutput(logits=values[0]))

model = BertForMaskedLM.from_pretrained('prajjwal1/bert-tiny')

BATCH_SIZE = 2
SEQ_LEN = 128
data = {
    'input_ids': torch.randint(30522, (BATCH_SIZE, SEQ_LEN)),
    # 'labels': torch.randint(30522, (BATCH_SIZE, SEQ_LEN)),
}
output = model(**data)


def mlir_compile(fx_g, inputs):
    for node in fx_g.graph.nodes:
        # TODO(byronyi): aten::t is not supported in DecomposeComplexOps
        if node.target == torch.ops.aten.t:
            fx_g.graph.inserting_after(node)
            new_node = fx_g.graph.call_function(
                torch.ops.aten.transpose, args=(node.args[0], 0, 1))
            node.replace_all_uses_with(new_node)
            fx_g.graph.erase_node(node)
        # TODO(byronyi): fx_g returning list breaks DecomposeComplexOps
        elif node.op == 'output':
            outputs = node.args
            num_outputs = len(node.args)
            node.args = (tuple(outputs) if num_outputs > 1 else outputs[0])
    fx_g.graph.lint()
    fx_g.recompile()

    module = torch_mlir.compile(
        fx_g, inputs, output_type=torch_mlir.OutputType.MHLO)
    fname = "bert_forward.mlir"
    with open(fname, "w+") as fout:
        fout.write(str(module))
        print("MHLO module has been save to {}".format(fname))
    print("MHLO execution is not support yet. Stopped.")
    exit(0)

    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)

    graph = torch.fx.Graph()
    args = [graph.placeholder(n.name)
            for n in fx_g.graph.nodes if n.op == 'placeholder']

    def execute(*args):
        rets = jit_module.forward(*[t.numpy() for t in args])
        return tuple([torch.from_numpy(t) for t in rets])
    graph.output(graph.call_function(execute, tuple(args)))
    graph.lint()
    return torch.fx.GraphModule(fx_g, graph)


decompositions = get_decompositions([
    torch.ops.aten.embedding,
])
compiled_model = aot_function(
    model, fw_compiler=mlir_compile, bw_compiler=nop, decompositions=decompositions)
compiled_output = compiled_model(**data)

for k in output:
    torch.testing.assert_close(
        output[k], compiled_output[k], atol=1e-4, rtol=1e-4)
