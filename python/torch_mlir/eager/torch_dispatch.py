import re
import warnings
from typing import Optional, List, Dict
from typing import cast, Tuple, Any, Callable

import torch
import torch._C
import torch.utils._pytree as pytree
from torch.fx.node import map_aggregate
from torch.fx.operator_schemas import (
    _args_kwargs_to_normalized_args_kwargs,
    type_matches,
    get_signature_for_torch_op,
    create_type_hint,
)
from torch.jit import ScriptFunction
from torchvision import models

from lazytensor.builder import _get_func_op_with_name
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)
from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report
from utils.annotator import Annotation, AnnotationConverter
from utils.torch_mlir_types import TorchTensorType

SUPPORTED_OPS = frozenset(
    [
        member.OPERATION_NAME
        for member in vars(torch_dialect).values()
        if hasattr(member, "OPERATION_NAME")
    ]
)


def check_supported_op(schema: torch._C.FunctionSchema) -> bool:
    return (
        "torch."
        + schema.name.replace("::", ".")
        + ("." + schema.overload_name if schema.overload_name else "")
    ) in SUPPORTED_OPS


def normalize_function(
    target: Callable, args: Tuple[Any], arg_types: Tuple[Any]
) -> Optional[Tuple[torch._C.FunctionSchema, Tuple[Any], Tuple[Any]]]:
    """
    This is basically torch.fx.operator_schemas.normalize_function but returns the schema as well as the
    new (normalized) args and (currently) no kwargs.

    It works basically by trying to bind values to type (using inspect.Signature) and then further disambiguating
    using arg_types.

    Raises RuntimeError if no unique, matching op schema can be identified.
    TODO: better error?
    """

    # the reference implementation handles kwargs but we don't yet.
    # leave this here as a placeholder for when we do (rather than
    # passing just bare {} in the rest of the body of this function)
    kwargs = {}
    kwarg_types = None
    normalize_to_only_use_kwargs = False

    # we don't handle target in boolean_dispatched or target.__module__ in ['torch.nn.functional', 'torch.functional']
    # we could (the reference implementation does)
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
            pass
        elif len(matches) == 1:
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                matches[0][0], args, kwargs, normalize_to_only_use_kwargs
            )
            return matches[0][1], new_args_and_kwargs.args, new_args_and_kwargs.kwargs
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
                        return (
                            candidate_schema,
                            new_args_and_kwargs.args,
                            new_args_and_kwargs.kwargs,
                        )
            else:
                # matched more than one schema despite providing types. Not sure if this is possible since in the reference
                # implementation the comment here states caller should provide types (optional for the reference impl)
                # where we require arg types.
                schema_printouts = "\n".join(str(schema) for schema in matches)
                raise RuntimeError(
                    f"Tried to normalize arguments to {torch.typename(target)} but "
                    f"the schema match was ambiguous! Please provide argument types to "
                    f"the normalize_arguments() call. Available schemas:\n{schema_printouts}"
                )

    raise RuntimeError("couldn't normalize args")


# shameless SO copy pasta
def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", "_", s)

    return s


def build_script_function(
    schema: torch._C.FunctionSchema,
    args: List[torch._C.Argument],
    kwargs: Dict[str, Any],
) -> torch.jit.ScriptFunction:
    """
    Build a torch.jit.ScriptFunction that corresponds to the schema.
    This is necessary so that we can pass a "hydrated" module to ModuleBuilder::importModule.
    Hydrated meaning with inlined constants with correct values; a gotcha here is that if you don't inline the constants
    then importModule assume default values (i think?) thereby (potentially) failing to support the op (not sure how this makes sense but native_batch_norm for example doesn't work without train=False)

    Works by
    1. creating an empty ts graph
    2. creating and inserting a node corresponding to the op; nodes inserted are just names (hence following steps)
    2. for each arg to the op:
        a. if arg is a tensor: add input to the graph corresponding to arg
           if arg is a constant (bool, int, list): inline a constant (at the top of the graph)
        b. wire value from previous step to the op (ie SSA value corresponding to graph input is "passed" to op)
        c. if arg is a tensor: fill in shape

    The (poorly handled/poorly understood) gotcha here is args with type Optional[Tensors] with values None.
    These pytorch types correspond to !torch.optional<!torch.vtensor>, which torch-mlir complains about having a torch.type_bound
    (something something torch.type_bound can only be a func arg_attr for torch.vtensor). The hacky solution here works but
    there's probably some straightforward way to handle this that I didn't stumble upon.

    Parameters
    ----------
    schema: torch._C.FunctionSchema (i.e., c10::FunctionSchema)
        Function in TorchScript IR to turn into MLIR.
    args: ...

    """
    graph = torch._C.Graph()
    node = graph.insertNode(graph.create(schema.name, len(schema.returns)))
    for i, arg in enumerate(schema.arguments):
        if arg.name in kwargs:
            val = kwargs[arg.name]
        else:
            val = args[i]

        if arg.type.isSubtypeOf(torch.TensorType.get()) or (
            isinstance(arg.type, torch.OptionalType) and val is not None
        ):
            inp = graph.addInput()
            if isinstance(arg.type, torch.OptionalType):
                inp.setType(arg.type.getElementType())
            else:
                inp.setType(arg.type)
        else:
            inp = graph.insertConstant(val)
            inp.node().moveBefore(node)

        node.addInput(inp)

    if node.hasMultipleOutputs():
        for outp in node.outputs():
            graph.registerOutput(outp)
    else:
        graph.registerOutput(node.output())

    # this name will ultimately be the name of the func op in the mlir module and thus needs to be unique
    # per op (since we keep the module around), hence smash together all of the things
    fn_name = urlify(str(node).strip())
    fn = torch._C._create_function_from_graph(fn_name, graph)
    return fn


def build_module(mb, jit_function: ScriptFunction, annotation: Annotation) -> ir.Module:
    func_op = _get_func_op_with_name(mb.module, jit_function.name)
    print("mlir op ir: \n", func_op)
    assert (
        func_op is not None
    ), "Unable to find FuncOp in new module. Make sure function was imported correctly into ModuleBuilder"

    arg_attrs = AnnotationConverter.to_mlir_array_attr(annotation, mb.context)
    func_op.attributes["arg_attrs"] = arg_attrs

    return mb.module


def try_torch_mlir_eager(func, args):
    print("torch op info: ", func.__module__, func.__name__)

    arg_types = map_aggregate(args, type)
    assert isinstance(arg_types, tuple)
    arg_type_hints = tuple([create_type_hint(i) for i in arg_types])
    schema, normalized_args, normalized_kwargs = normalize_function(
        func, args, arg_types=arg_type_hints
    )
    if not check_supported_op(schema):
        warnings.warn(
            f"{schema.name}.{schema.overload_name} not supported in TorchMLIR yet."
        )
        return None

    script_fun = build_script_function(schema, normalized_args, normalized_kwargs)

    annotations = []
    tensor_args = []
    for i, arg in enumerate(normalized_args):
        if isinstance(arg, torch.Tensor) or isinstance(
            arg, torch.nn.parameter.Parameter
        ):
            annotations.append(TorchTensorType(shape=tuple(arg.shape), dtype=arg.dtype))
            tensor_args.append(arg.numpy())
    func_annotation = Annotation(annotations)
    mb = ModuleBuilder()
    mb.import_function(script_fun)
    eager_module = build_module(mb, script_fun, func_annotation)

    run_pipeline_with_repro_report(
        eager_module,
        "torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline",
        func.__name__,
    )
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(eager_module)
    jit_module = backend.load(compiled)

    numpy_rs = getattr(jit_module, script_fun.name)(*tensor_args)
    torch_rs = pytree.tree_map(torch.from_numpy, numpy_rs)
    return torch_rs


class TorchMLIRTensor(torch.Tensor):
    elem: torch.Tensor
    __backend = RefBackendLinalgOnTensorsBackend()
    __slots__ = ["elem"]

    @classmethod
    def backend(cls):
        return cls.__backend

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
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, TorchMLIRTensor) else e

        def wrap(e):
            return TorchMLIRTensor(e) if isinstance(e, torch.Tensor) else e

        args = pytree.tree_map(unwrap, args)
        kwargs = pytree.tree_map(unwrap, kwargs)
        if kwargs:
            warnings.warn(
                f"TorchMLIR doesn't yet support passing kwargs; running through PyTorch eager."
            )
            return pytree.tree_map(wrap, func(*args, **pytree.tree_map(unwrap, kwargs)))

        res = try_torch_mlir_eager(func, args)
        if res is None:
            warnings.warn("Couldn't compile using TorchMLIR; running through PyTorch eager.")
            return pytree.tree_map(wrap, func(*args, **kwargs))
        else:
            return pytree.tree_map(wrap, res)


def turn_off_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


def test_resnet():
    mod = ResNet18Module()
    turn_off_gradients(mod)
    t = TorchMLIRTensor(torch.randn((1, 3, 32, 32), requires_grad=False))
    y = mod(t)
    # loss = y.sum()
    # for i in range(5):
    #     print("*" * 10)
    # loss.backward()
    print(y)


if __name__ == "__main__":
    with torch.no_grad():
        test_resnet()

