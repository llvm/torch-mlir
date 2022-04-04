import torch
import torchvision

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, \
    ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import \
    extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import \
    LinalgOnTensorsTosaBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import \
    RefBackendLinalgOnTensorsBackend


class ControlFlowTestModule(torch.nn.Module):
    def __init__(self, do_activation: bool = False):
        super().__init__()
        self.do_activation = do_activation

    @export
    @annotate_args([
        None,
        ([2, 2], torch.float32, True),
        ([2, 2], torch.float32, True),
        ([], torch.int32, True),
        ([], torch.int32, True),
    ])
    def forward(self, x, y, a, b):
        # if-control
        if a == b:
            x = x - y
        else:
            x = x + y
        return torch.add(x, y)
        # for-loop
        # x = x + y
        # for i in range(5):
        #     x = x - y
        # t = torch.zeros(2, 2)
        # for i in range(12):
        #     for j in range(12):
        #         # There could be many loops here, or if statements, etc.
        #         t = t.t()
        # return torch.add(x, y)
        # while-loop
        # while(x[0] > y[0]):
        #     x = x - y
        # return torch.add(x, y)

        # # multi-output
        # z = x + y
        # if x == y:
        #     x = x - y
        #     z = x + y
        # else:
        #     x = x + y
        #     z = x - y
        # return torch.add(x, z)


BACKEND = RefBackendLinalgOnTensorsBackend()


# BACKEND = LinalgOnTensorsTosaBackend()


def compile_module(program: torch.nn.Module):
    """Compiles a torch.nn.Module into an compiled artifact.
    This artifact is suitable for inclusion in a user's application. It only
    depends on the rebackend runtime.
    """
    ## Script the program.
    scripted = torch.jit.script(program)
    print("-------------------------------graph-------------------------------")
    print(scripted.graph)
    print("-------------------------------code-------------------------------")
    print(scripted.code)
    print("-------------------------------_c-------------------------------")
    print(scripted._c.code)

    ## Extract annotations.
    class_annotator = ClassAnnotator()
    extract_annotations(program, scripted, class_annotator)

    ## Import the TorchScript module into MLIR.
    mb = ModuleBuilder()
    mb.import_module(scripted._c, class_annotator)
    print(
        "-------------------------------init IR-------------------------------")
    mb.module.dump()

    ## Lower the MLIR from TorchScript to RefBackend, passing through linalg-on-tensors.
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline',
                           mb.module.context)
    pm.run(mb.module)
    print(
        "-------------------------------torch-back-pipeline-------------------------------")
    mb.module.dump()

    pm = PassManager.parse(
        'torch-backend-to-linalg-on-tensors-backend-pipeline',
        mb.module.context)
    pm.run(mb.module)
    print(
        "-------------------------------linalg-------------------------------")
    mb.module.dump()

    ## Invoke RefBackend to compile to compiled artifact form.
    compiled_module = BACKEND.compile(mb.module)
    print(
        "-------------------------------after-compile-------------------------------")
    compiled_module.dump()
    return compiled_module


if __name__ == '__main__':
    print("---------------main-------------------")
    mlp_module = ControlFlowTestModule(do_activation=False)
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    # Run it!
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    y = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    a = torch.scalar_tensor(3)
    b = torch.scalar_tensor(2)
    print("x: ", x, "\n\n", "y: ", y)
    print("\nx+2y = ")
    print("\n\ngolden result: ", mlp_module.forward(x, y, a, b))
    print("\n\ntorch-mlir result: ", jit_module.forward(x.numpy(), y.numpy(), a.numpy(),
                                                        b.numpy()))