import torch
from torch_mlir import torchscript

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

class Inner(object):
    # CHECK-LABEL: func.func private @__torch__.Inner.foo(
    # CHECK-SAME:      %[[ARG:.*]]: !torch.nn.Module<"__torch__.Inner">) {
    # CHECK:         torch.constant.int 42
    # CHECK:         torch.prim.Store "cls", %[[ARG]] : !torch.nn.Module<"__torch__.Inner">
    # CHECK:         %[[DICT:.*]] = torch.prim.DictConstruct keys() values() -> !torch.dict<str, tensor>
    # CHECK:         torch.prim.Store "this_dict", %[[DICT]] : !torch.dict<str, tensor>
    # CHECK:         torch.prim.Load "this_dict" : !torch.dict<str, tensor>
    # CHECK:         torch.constant.str "key"
    # CHECK:         return
    # CHECK:       }

    @classmethod
    def foo(cls):
        this_dict = {}
        this_dict["key"] = 42
        return this_dict


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()

    # CHECK-LABEL: func.func private @__torch__.Model.forward
    # CHECK-SAME:      (%[[module:.*]]: !torch.nn.Module<"__torch__.Model">, %[[tensor:.*]]: !torch.tensor
    # CHECK:         %[[object:.*]] = torch.prim.CreateObject !torch.nn.Module<"__torch__.torch.autograd.grad_mode.no_grad">
    # CHECK:         %[[init:.*]] = torch.prim.CallMethod %1["__init__"] () : !torch.nn.Module<"__torch__.torch.autograd.grad_mode.no_grad">, () -> !torch.none
    # CHECK:         %[[enter:.*]] = torch.prim.Enter %[[object]] : !torch.nn.Module<"__torch__.torch.autograd.grad_mode.no_grad">
    # CHECK:         %[[exit:.*]] = torch.prim.Exit %[[object]] : !torch.nn.Module<"__torch__.torch.autograd.grad_mode.no_grad"> -> !torch.tensor
    # CHECK:         return %[[tensor]] : !torch.tensor

    def forward(self, data):
        with torch.no_grad():
            return data

output_type = torchscript.OutputType.RAW
mod = torchscript.compile(Model(), [torch.tensor([0, 1, 2, 3])], output_type)
print(mod)
