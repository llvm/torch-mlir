import io
import re
from functools import lru_cache
from typing import Optional, List
from typing import cast, Tuple, Any, Callable, Dict

import torch
import torch._C
import torch.utils._pytree as pytree
from torch._C import parse_schema
from torch._sources import get_source_lines_and_file
from torch.fx.node import map_aggregate
from torch.fx.operator_schemas import (
    _args_kwargs_to_normalized_args_kwargs,
    type_matches,
    get_signature_for_torch_op,
    create_type_hint,
)
from torch.jit.frontend import get_jit_class_def
from torch.testing._internal.common_utils import TemporaryFileName
from torch_mlir import ir
from torch_mlir.dialects.builtin import FuncOp
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder, ClassAnnotator
from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)
from torchvision import models

from examples.lazytensor.builder import build_module
from examples.utils.annotator import Annotation
from examples.utils.torch_mlir_types import TorchTensorType, PythonType
from python.torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from python.torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from python.torch_mlir_e2e_test.utils import run_pipeline_with_repro_report


def get_func_op_with_name(module: ir.Module, name: str) -> Optional[FuncOp]:
    @lru_cache
    def find_in_ops(name_attr):
        for op in module.body.operations:
            if isinstance(op, FuncOp) and op.name.value == name_attr:
                return op
        return None

    with module.context:
        name_attr = ir.StringAttr.get(name).value

    return find_in_ops(name_attr)


def normalize_function(
        target: Callable,
        args: Tuple[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        arg_types: Optional[Tuple[Any]] = None,
        kwarg_types: Optional[Dict[str, Any]] = None,
        normalize_to_only_use_kwargs: bool = False,
):
    if kwargs is None:
        kwargs = {}

    # we don't handle target in boolean_dispatched or target.__module__ in ['torch.nn.functional', 'torch.functional']
    assert callable(target)
    torch_op_signatures, torch_op_schemas = get_signature_for_torch_op(
        target, return_schemas=True
    )
    matches = []
    if torch_op_signatures:
        for candidate_signature, candidate_schema in zip(
                torch_op_signatures, torch_op_schemas
        ):
            try:
                candidate_signature.bind(*args, **kwargs)
                matches.append((candidate_signature, candidate_schema))
            except TypeError as e:
                continue

        if len(matches) == 0:
            # Did not match any schema. Cannot normalize
            pass
        elif len(matches) == 1:
            # Matched exactly one schema, unambiguous
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                matches[0][0], args, kwargs, normalize_to_only_use_kwargs
            )
            # return the schema (we don't care about sig?)
            return matches[0][1], new_args_and_kwargs
        else:
            if arg_types is not None or kwarg_types is not None:
                arg_types = arg_types if arg_types else cast(Tuple[Any], ())
                kwarg_types = kwarg_types if kwarg_types else {}
                for candidate_signature, candidate_schema in matches:
                    sig_matches = True
                    try:
                        bound_types = candidate_signature.bind(
                            *arg_types, **kwarg_types
                        )
                        for arg_name, arg_type in bound_types.arguments.items():
                            param = candidate_signature.parameters[arg_name]
                            sig_matches = sig_matches and type_matches(
                                param.annotation, arg_type
                            )
                    except TypeError as e:
                        sig_matches = False
                    if sig_matches:
                        new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                            candidate_signature,
                            args,
                            kwargs,
                            normalize_to_only_use_kwargs,
                        )
                        return candidate_schema, new_args_and_kwargs
            else:
                # Matched more than one schema. In this situation, the caller must provide the types of
                # the arguments of the overload they expect.
                schema_printouts = "\n".join(str(schema) for schema in matches)
                raise RuntimeError(
                    f"Tried to normalize arguments to {torch.typename(target)} but "
                    f"the schema match was ambiguous! Please provide argument types to "
                    f"the normalize_arguments() call. Available schemas:\n{schema_printouts}"
                )

    raise RuntimeError("couldn't normalize args")


def get_types(args, kwargs):
    def get_type(arg):
        return type(arg)

    arg_types = map_aggregate(args, get_type)
    assert isinstance(arg_types, tuple)
    arg_types = tuple([create_type_hint(i) for i in arg_types])
    kwarg_types = {k: get_type(v) for k, v in kwargs.items()}

    return arg_types, kwarg_types


def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", "_", s)

    return s


def getExportImportCopy(m, also_test_file=True, map_location=None):
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    buffer.seek(0)
    imported = torch.jit.load(buffer, map_location=map_location)

    if not also_test_file:
        return imported

    with TemporaryFileName() as fname:
        torch.jit.save(imported, fname)
        return torch.jit.load(fname, map_location=map_location)


def createFunctionFromGraph(trace):
    graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
    fn = torch._C._create_function_from_graph("forward", graph)
    m_import = getExportImportCopy(fn)
    print(m_import)


def make_strong_fn_ptr(schema_str, args, kwargs):
    schema = parse_schema(schema_str)
    graph = torch._C.Graph()
    node = graph.insertNode(graph.create(schema.name, len(schema.returns)))
    for i, arg in enumerate(schema.arguments):
        if not (arg.type.isSubtypeOf(torch.TensorType.get()) or "Optional" in arg.type.annotation_str):
            inp = graph.insertConstant(args[i])
            inp.node().moveBefore(node)
        else:
            inp = graph.addInput()
            if "Optional" in arg.type.annotation_str and args[i] is not None:
                inp.setType(arg.type.getElementType())
            else:
                inp.setType(arg.type)

        node.addInput(inp)

    if node.hasMultipleOutputs():
        for outp in node.outputs():
            graph.registerOutput(outp)
    else:
        graph.registerOutput(node.output())

    fn_name = urlify(str(node).strip())
    fn = torch._C._create_function_from_graph(fn_name, graph)
    return fn


backend = RefBackendLinalgOnTensorsBackend()


def create_with_class_annotator(mb, recursivescriptmodule):
    class_annotator = ClassAnnotator()

    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
            None,
            (None, torch.int, True),
            (None, torch.int, True),
            (None, torch.int, True),
            (None, torch.bool, True),
            (None, torch.int, True),
            (None, torch.int, True),
        ],
    )
    mb.import_module(recursivescriptmodule._c, class_annotator)


class TorchMLIRTensor(torch.Tensor):
    elem: torch.Tensor
    __module_builder: ModuleBuilder = ModuleBuilder()
    __slots__ = ["elem"]

    @staticmethod
    def module_builder(cls):
        return cls.__module_builder

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=elem.device,
        )
        r.elem = elem
        return r

    def __repr__(self):
        return f"TorchMLIRTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, TorchMLIRTensor) else e

        def wrap(e):
            return TorchMLIRTensor(e) if isinstance(e, torch.Tensor) else e

        args = pytree.tree_map(unwrap, args)
        kwargs = pytree.tree_map(unwrap, kwargs)
        arg_types, kwarg_types = get_types(args, kwargs)
        schema, (new_args, new_kwargs) = normalize_function(
            func, args, kwargs, arg_types=arg_types, kwarg_types=kwarg_types
        )

        mb = cls.module_builder(cls)
        str_ptr = make_strong_fn_ptr(str(schema), new_args, new_kwargs)
        mb.import_function(str_ptr)

        op = get_func_op_with_name(mb.module, str_ptr.name)
        print("torch op info: ", func.__module__, func.__name__)
        print("mlir op ir: \n", op)

        if "torch.operator" not in str(op):
            annotations = []
            tensor_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    annotations.append(
                        TorchTensorType(shape=tuple(arg.shape), dtype=torch.float32)
                    )
                    tensor_args.append(arg.detach().numpy())
                else:
                    annotations.append(None)

            func_annotation = Annotation(annotations)
            eager_module = build_module(str_ptr, func_annotation)
            run_pipeline_with_repro_report(
                eager_module,
                "torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline",
                func.__name__,
            )
            backend = refbackend.RefBackendLinalgOnTensorsBackend()
            compiled = backend.compile(eager_module)
            jit_module = backend.load(compiled)
            rs = getattr(jit_module, op.name.value)(*tensor_args)
            print("result: ", rs)
            print()
        else:
            rs = pytree.tree_map(wrap, func(*args, **pytree.tree_map(unwrap, kwargs)))

        return rs


def f(x):
    if x[0] % 2:
        y = x.sin()
    else:
        y = x.cos()
    return y


def g(x):
    y = x.sin()
    z = y.cos()
    w = z.sum()
    return w


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


class Conv2dNoPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=True)
        self.train(False)

    def forward(self, x):
        return self.conv(x)


class NativeBatchNorm1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
            ([-1], torch.float32, True),
            ([-1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten.native_batch_norm(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            training=False,
            momentum=0.1,
            eps=0.00001,
        )


def test_batch_norm():
    mb = ModuleBuilder()
    test_module = NativeBatchNorm1DModule()
    recursivescriptmodule = torch.jit.script(test_module)
    torch._C._jit_pass_inline(recursivescriptmodule.graph)
    print(recursivescriptmodule.graph)
    str_ptr = torch._C._create_function_from_graph("test", recursivescriptmodule.graph)
    mb.import_function(str_ptr)

    # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
    # mb.import_module(recursivescriptmodule._c)
    mb.module.operation.print()
    mlir_module = mb.module

    run_pipeline_with_repro_report(
        mlir_module,
        "torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline",
        "",
    )
    compiled = backend.compile(mlir_module)
    jit_module = backend.load(compiled)


def test_conv2d():
    mod = Conv2dNoPaddingModule()
    t = TorchMLIRTensor(torch.rand(5, 2, 10, 20))
    y = mod(t)
    print(y)


def test_resnet():
    mod = ResNet18Module()
    t = TorchMLIRTensor(torch.randn((1, 3, 32, 32), requires_grad=True))
    # print(t)
    y = mod(t)
    loss = y.sum()
    for i in range(5):
        print("*" * 10)
    loss.backward()


class MyClass:
    shared_variable = 3

    def __init__(self):
        if self.shared_variable is None:
            raise RuntimeError("shared_variable must be set before instantiating")


if __name__ == "__main__":
    # test_conv2d()
    test_resnet()
    # test_batch_norm()
    # objA = MyClass()
    # print(objA.shared_variable)
    # objA.shared_variable = 5
    # print(objA.shared_variable)
    # objB = MyClass()
    # print(objB.shared_variable)

# func_annotation = Annotation([TorchTensorType(shape=shape, dtype=torch.float),
#                               TorchTensorType(shape=shape, dtype=torch.float)])
# mlir_module = build_module(script_function, func_annotation)
#
# print("MLIR")
# mlir_module.dump()
#
# # Compile the torch MLIR and execute the compiled program
#
# print("BEFORE LINALG-ON-TENSORS BACKEND PIPELINE")
# print(mlir_module)
#
# backend = RefBackendLinalgOnTensorsBackend()
# compiled = backend.compile(mlir_module)
# jit_module = backend.load(compiled)
#
# print("\n\nRunning Example Calculation")
# print("Compiled result:")
# print(jit_module.my_method(x.cpu().numpy(), y.cpu().numpy()))
# print("Expected result:")
# print(computation(x, y))
