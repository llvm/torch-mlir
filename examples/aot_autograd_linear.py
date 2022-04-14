import torch

from functorch.compile import aot_module
from functorch.compile import get_decompositions

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

_CHECK_MHLO = True

model = torch.nn.Linear(3, 4)

# TODO: none in output breaks AdjustCallingConventions
data = torch.rand((2, 3)).requires_grad_()
output = model(data)


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
            node.args = (tuple(node.args[0]),)
    fx_g.graph.lint()
    fx_g.recompile()

    if _CHECK_MHLO:
        module = torch_mlir.compile(
            fx_g, inputs, output_type=torch_mlir.OutputType.MHLO)
        print(module)
        exit(0)
    module = torch_mlir.compile(
        fx_g, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
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
    torch.ops.aten.detach,
])
compiled_model = aot_module(
    model, mlir_compile, decompositions=decompositions)
compiled_output = compiled_model(data)

torch.testing.assert_close(output, compiled_output)

output.sum().backward()
grads = {k: torch.clone(v.grad) for k, v in model.named_parameters()}
grads['data'] = torch.clone(data.grad)
data.grad.zero_()
model.zero_grad()

compiled_output.sum().backward()
compiled_grads = {k: torch.clone(v.grad) for k, v in model.named_parameters()}
compiled_grads['data'] = torch.clone(data.grad)
data.grad.zero_()
model.zero_grad()

for k in grads:
    torch.testing.assert_close(grads[k], compiled_grads[k])
