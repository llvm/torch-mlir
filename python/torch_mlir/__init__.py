# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Sequence, Union, List
from enum import Enum

import sys
from io import StringIO

import torch

from torch_mlir.passmanager import PassManager
from .compiler_utils import run_pipeline_with_repro_report
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ImportOptions, ModuleBuilder


class OutputType(Enum):
    """The kind of output that `torch_mlir.compile` can produce.

    In MLIR terminology, this describes the mix of dialects that will be
    produced by the conversion process.

    In user-facing API's, this type can always be passed interchangeably with an
    appropriate string specifying the output type. The allowed strings are
    the set of enum vales, allowed to be case insensitive and with `-` allowed
    in place of `_`. The `OutputType.get` static method can be used to convert
    from a string to an `OutputType` instance.
    """

    # This output type consists of `torch` dialect ops that have been converted
    # maximally to value semantics, decomposed, and shapes have been inferred.
    TORCH = "torch"

    # The output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = "linalg-on-tensors"

    # This output type consists of `tosa` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to TOSA.
    TOSA = "tosa"

    # This output type consists of `mhlo` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to MHLO.
    MHLO = "mhlo"

    # Raw output of the JIT IR importer. This is not expected to be useful
    # for end-users, but can be convenient for development or reporting bugs.
    RAW = "raw"

    @staticmethod
    def get(spec: Union[str, "OutputType"]) -> "OutputType":
        """Gets an OutputType from allowed way to specify one.

        Args:
          spec: An OutputType instance or the case-insensitive name of one of the
            enum values.
        Returns:
          An OutputType instance.
        """
        if isinstance(spec, OutputType):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in OutputType.__members__:
            raise ValueError(f"For output_type= argument, expected one of: "
                             f"{', '.join(OutputType.__members__.keys())}")
        return OutputType[spec]



class TensorPlaceholder:
    """A class that represents a formal parameter of a given shape and dtype.

    This class can be constructed explicitly from a shape and dtype:
    ```python
    placeholder = TensorPlaceholder([3, 4], torch.float32)
    ```

    This class can also be constructed from a `torch.Tensor` which is already
    known to be a valid input to the function. In this case, a set of
    dynamic axes are allowed to be specified.
    ```python
    placeholder = TensorPlaceholder.like(torch.ones(3, 4), dynamic_axes=[1])
    # Equivalent to `TensorPlaceholder([3, -1], torch.float32)`
    ```
    """

    def __init__(self, shape: List[int], dtype: torch.dtype):
        """Create a tensor with shape `shape` and dtype `dtype`.

        Args:
            shape: The shape of the tensor. A size of `-1` indicates that the
            dimension has an unknown size.
            dtype: The dtype of the tensor.
        """
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def like(tensor: torch.Tensor, dynamic_axes: List[int] = None):
        """Create a tensor placeholder that is like the given tensor.

        Args:
            tensor: The tensor to create a placeholder for.
            dynamic_axes: A list of dynamic axes. If specified, the compiled
            module will allow those axes to be any size at runtime.
        """
        if dynamic_axes is None:
            dynamic_axes = []
        shape = []
        for i, dim in enumerate(tensor.shape):
            if i in dynamic_axes:
                shape.append(-1)
            else:
                shape.append(dim)
        return TensorPlaceholder(shape, tensor.dtype)


# The set of ops that are considered legal for each backend.
# These are currently quite load-bearing, since different backends might be
# missing patterns for decomposed forms of certain ops.
# TODO: Tighten up the definition of these "conditionally legal for backends"
# ops in the backend contract, and move these lists somewhere deeper in the
# compiler where each backend can "own" its set of legal ops.
BACKEND_LEGAL_OPS = {
    OutputType.TOSA: ['torch.aten.flatten.using_ints','torch.aten.native_layer_norm','torch.aten.linear'],
    OutputType.LINALG_ON_TENSORS: ['torch.aten.flatten.using_ints',],
    OutputType.MHLO: [],
}


_example_arg = Union[TensorPlaceholder, torch.Tensor]


def compile(model: torch.nn.Module,
            example_args: Union[_example_arg, Sequence[_example_arg]],
            output_type: Union[str, "OutputType"] = OutputType.TORCH,
            use_tracing: bool = False,
            ignore_traced_shapes = False,
            verbose: bool = False):
    """Convert a PyTorch model to MLIR.

    Args:
        model: The PyTorch model to convert.
        example_args: A list of example arguments to use when inferring the
            shapes of the arguments to `forward` method of the model.
            A single tensor is treated as a list of a single tensor.
            A TensorPlaceholder object is also allowed in the place of any
            Tensor.
        output_type: The kind of output to produce. See `OutputType` for more
            details.
        use_tracing: If True, use `torch.jit.trace` to convert the model to
            JIT IR rather than `torch.jit.script`.
        ignore_traced_shapes: If True, ignore the shapes that were observed
            during tracing. This should only be used if one knows that the
            original traced program would result in the same trace (modulo
            shapes) for all shape combinations implied by any
            `TensorPlaceholder`'s used as `example_args`. Also,
            strictly-speaking, this option covers dtypes too, but we just say
            "shapes" to be succinct.
        verbose: If true, print extra information about the conversion.

    Returns:
        An MLIR module that contains the converted model in the specified
        output type.
    """
    output_type = OutputType.get(output_type)
    if ignore_traced_shapes and not use_tracing:
        raise Exception("`ignore_traced_shapes` requires `use_tracing`")

    # Special case -- many models have just one input, so canonicalize a single
    # tensor to a list of a single tensor to make the API more ergonomic.
    if isinstance(example_args, (torch.Tensor, TensorPlaceholder)):
        example_args = (example_args,)
    
    # If users passed in anything other than tensors or a list of tensors (e.g.
    # a dictionary), we can't handle it.
    if not isinstance(example_args, Sequence):
        raise Exception(
            "Only Tensors, TensorPlaceholders, or a sequences of Tensors and "
            "TensorPlaceholders are supported as inputs.")

    # TODO: Don't hardcode "forward". See `torch.onnx.export` and
    # `torch.jit.trace_module` for API inspiration.
    if use_tracing:
        example_args_for_trace = []
        for arg in example_args:
            if isinstance(arg, TensorPlaceholder):
                if not ignore_traced_shapes:
                    # To avoid accidental footguns, we require
                    # `ignore_traced_shapes` to be true if we're using
                    # TensorPlaceholder's, as it falls into the same
                    # "hopefully the trace works for different inputs" bucket
                    # of concerns.
                    raise Exception(
                        "TensorPlaceholder can only be used with tracing when `ignore_traced_shapes=True`")
                # For any dynamic dimensions, replace them with "7" arbitrarily.
                # If a user is using dynamic dimensions with tracing, they are
                # walking on thin ice already -- assume they know what they are
                # doing.
                shape = [s if s != -1 else 7 for s in arg.shape]
                example_args_for_trace.append(
                    torch.ones(*shape, dtype=arg.dtype))
            elif isinstance(arg, torch.Tensor):
                example_args_for_trace.append(arg)
            else:
                raise Exception(
                    "Only Tensors, TensorPlaceholders, or a sequences of "
                    "Tensors and TensorPlaceholders are supported as inputs.")
        scripted = torch.jit.trace(model, tuple(example_args_for_trace))
    else:
        scripted = torch.jit.script(model)
    # Convert all concrete inputs to TensorPlaceholder's, for consistency.
    arg_placeholders = []
    for arg in example_args:
        if isinstance(arg, TensorPlaceholder):
            arg_placeholders.append(arg)
        else:
            assert isinstance(arg, torch.Tensor)
            arg_placeholders.append(TensorPlaceholder.like(arg))

    class_annotator = ClassAnnotator()
    forward_annotation = [None]
    for arg in arg_placeholders:
        # Assume that all tensors have value semantics for now.
        forward_annotation.append((arg.shape, arg.dtype, True))
    class_annotator.exportNone(scripted._c._type())
    class_annotator.exportPath(scripted._c._type(), ["forward"])
    class_annotator.annotateArgs(
        scripted._c._type(), ["forward"], forward_annotation)

    mb = ModuleBuilder()
    import_options = ImportOptions()
    import_options.ignoreExistingTensorShapesAndDtypes = ignore_traced_shapes
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Import the TorchScript module to MLIR
        mb.import_module(scripted._c, class_annotator, import_options)
    except Exception as e:
        raise Exception(f"""
PyTorch TorchScript module -> torch-mlir Object Graph IR import failed with:
### Importer C++ Exception:
{e}
### Importer Diagnostics:
{sys.stderr.getvalue()}
""") from None
    finally:
        sys.stderr = original_stderr
    if output_type == OutputType.RAW:
        return mb.module

    backend_legal_ops = BACKEND_LEGAL_OPS.get(output_type, [])
    option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
    run_pipeline_with_repro_report(
        mb.module,
        f"torchscript-module-to-torch-backend-pipeline{option_string}",
        "Lowering TorchScript IR -> Torch Backend IR",
    )

    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(mb.module)

    if output_type == OutputType.TORCH:
        return mb.module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-tosa-backend-pipeline",
            "Lowering Torch Backend IR -> TOSA Backend IR")
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(mb.module)
        return mb.module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-linalg-on-tensors-backend-pipeline",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR")
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(mb.module)
        return mb.module

    elif output_type == OutputType.MHLO:
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-mhlo-backend-pipeline",
            "Lowering Torch Backend IR -> MHLO Backend IR")
        if verbose:
            print("\n====================")
            print("MHLO Backend IR")
            print(mb.module)
        return mb.module
    raise Exception(f"Unknown OutputType: {output_type}")
