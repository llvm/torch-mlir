import torch
import torch.utils.cpp_extension
import torch_mlir
from torch_mlir import run_pipeline_with_repro_report
from torch_mlir.ir import BoolAttr, Context, Module, InsertionPoint, Location
from torch_mlir_e2e_test.annotations import export, annotate_args


def identity(_5: torch.Tensor):
    return _5


goofy_lib = torch.library.Library("goofy", "DEF")
goofy_lib.define("identity(Tensor t) -> Tensor")
goofy_lib.impl("identity", identity)


class CustomOpExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        b = 2 * a
        return torch.ops.goofy.identity(b)


mod = CustomOpExampleModule()
mod.eval()

module = torch_mlir.compile(mod, torch.ones(3, 4), output_type="raw")

pipeline = (
    "symbol-dce,"
    "torch-prepare-for-globalize-object-graph,"
    "torch-globalize-object-graph,"
    "symbol-dce,"
    "inline{default-pipeline= max-iterations=4 },"
    "torch-adjust-calling-conventions"
)

run_pipeline_with_repro_report(
    module, pipeline=f"builtin.module({pipeline})", description=""
)
print(module)

forward = module.operation.regions[0].blocks[0].operations[1]
goofy_op = forward.operation.regions[0].blocks[0].operations[4]
goofy_op.attributes["has_value_semantics"] = BoolAttr.get(True, context=module.context)

print(module)

abstract_interp_src = """\
func.func @__torch_mlir_shape_fn.operator.goofy.identity(%arg0: !torch.list<int>) -> !torch.list<int> {
  return %arg0 : !torch.list<int>
}
func.func @__torch_mlir_dtype_fn.operator.goofy.identity(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  return %arg1 : !torch.int
}
"""

with Location.unknown(module.context) as loc:
    abstract_interp_module = Module.parse(abstract_interp_src)
    with InsertionPoint.at_block_begin(module.body) as ip:
        shape_fn = abstract_interp_module.body.operations[0]
        dtype_fn = abstract_interp_module.body.operations[1]
        InsertionPoint.insert(ip, shape_fn.detach_from_parent())
        InsertionPoint.insert(ip, dtype_fn.detach_from_parent())

print(module)

run_pipeline_with_repro_report(
    module,
    pipeline="builtin.module(func.func(torch-reduce-op-variants,torch-maximize-value-semantics))",
    description="",
)

print(module)

run_pipeline_with_repro_report(
    module,
    pipeline="builtin.module(torch-lower-to-backend-contract{backend-legal-ops=torch.operator decompose=true max-iterations=10})",
    description="",
)

shape_fn.detach_from_parent()
dtype_fn.detach_from_parent()

print(module)
