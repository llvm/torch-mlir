import os
import tempfile
from typing import List, Tuple

import torch
import torch.multiprocessing as mp
import torch.utils.cpp_extension
from torch_mlir import torchscript
from torch_mlir_e2e_test.annotations import export, annotate_args


# RUN: %PYTHON %s | FileCheck %s


def identity(x: torch.Tensor):
    return x


goofy_lib = torch.library.Library("goofy", "DEF")
goofy_lib.define("identity(Tensor t) -> Tensor")
goofy_lib.impl("identity", identity)

def goofy〇identity〡shape(t: List[int]) -> List[int]:
    return t

def goofy〇identity〡dtype(t_rank_dtype: Tuple[int, int]) -> int:
    t_rank, t_dtype = t_rank_dtype
    return t_dtype

def goofy〇identity〡has_value_semantics() -> None:
    return

extra_library = [
    goofy〇identity〡shape, goofy〇identity〡dtype, goofy〇identity〡has_value_semantics]

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

def run():
    mod = CustomOpExampleModule()
    mod.eval()

    module = torchscript.compile(
        mod,
        torch.ones(3, 4),
        output_type="torch",
        backend_legal_ops=["goofy.identity"],
        extra_library=extra_library,
    )

    print(module)

run()

# CHECK:    module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
# CHECK:      func.func @forward(%{{.*}}: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
# CHECK:        %{{.*}} = torch.constant.int 2
# CHECK:        %{{.*}} = torch.aten.mul.Scalar %{{.*}}, %{{.*}} : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
# CHECK:        %{{.*}} = torch.operator "goofy.identity"(%{{.*}}) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK:        return %1 : !torch.vtensor<[3,4],f32>
# CHECK:      }
# CHECK:    }

# Using `torch.multiprocessing` adds extra namespaces to the abstract
# interpretation functions when they are imported into MLIR:
#   `func @"__torch__.__mp_main__.{name}...`
# This tests that the extra namespaces are removed correctly.
if __name__ == "__main__":
    mp.set_start_method("spawn")
    p = mp.Process(target=run, args=())
    p.start()
    p.join()

# CHECK:    module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
# CHECK:      func.func @forward(%{{.*}}: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
# CHECK:        %{{.*}} = torch.constant.int 2
# CHECK:        %{{.*}} = torch.aten.mul.Scalar %{{.*}}, %{{.*}} : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
# CHECK:        %{{.*}} = torch.operator "goofy.identity"(%{{.*}}) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK:        return %1 : !torch.vtensor<[3,4],f32>
# CHECK:      }
# CHECK:    }
