import os
import tempfile

import torch
import torch.utils.cpp_extension
import torch_mlir
from torch_mlir_e2e_test.annotations import export, annotate_args


# RUN: %PYTHON %s | FileCheck %s


def identity(x: torch.Tensor):
    return x


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

abstract_interp_src = """\
func.func @__torch_mlir_shape_fn.goofy.identity(%arg0: !torch.list<int>) -> !torch.list<int> {
  return %arg0 : !torch.list<int>
}
func.func @__torch_mlir_dtype_fn.goofy.identity(%arg0 : !torch.tuple<int, int>) -> !torch.int {
  %0:2 = torch.prim.TupleUnpack %arg0 : !torch.tuple<int, int> -> !torch.int, !torch.int
  return %0#1 : !torch.int
}
func.func @__torch_mlir_has_value_semantics_fn.goofy.identity() { return }
"""

with open("/tmp/custom_op_shape_dtype_fn.mlir", "w") as tmp:
    tmp.write(abstract_interp_src)

module = torch_mlir.compile(
    mod,
    torch.ones(3, 4),
    output_type="torch",
    backend_legal_ops=["goofy.identity"],
    _completely_unsupported_in_progress_extra_library="/tmp/custom_op_shape_dtype_fn.mlir",
)

print(module)

# CHECK:    module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
# CHECK:      func.func @forward(%{{.*}}: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
# CHECK:        %{{.*}} = torch.constant.int 2
# CHECK:        %{{.*}} = torch.aten.mul.Scalar %{{.*}}, %{{.*}} : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
# CHECK:        %{{.*}} = torch.operator "goofy.identity"(%{{.*}}) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK:        return %1 : !torch.vtensor<[3,4],f32>
# CHECK:      }
# CHECK:    }
