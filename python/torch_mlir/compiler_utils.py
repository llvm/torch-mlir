# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
from enum import Enum
from io import StringIO
import os
import sys
import tempfile
from typing import Union, List

import torch
from .passmanager import PassManager
from .ir import StringAttr


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


def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value


class TorchMlirCompilerError(Exception):
    pass


def run_pipeline_with_repro_report(
    module, pipeline: str, description: str, enable_ir_printing: bool = False
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    original_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context as ctx:
            # TODO(#3506): Passes can emit errors but not signal failure,
            # which causes a native assert.
            ctx.emit_error_diagnostics = True
            pm = PassManager.parse(pipeline)
            if enable_ir_printing:
                ctx.enable_multithreading(False)
                pm.enable_ir_printing()
            pm.run(module.operation)
    except Exception as e:
        # TODO: More robust.
        # - don't arbitrarily clutter up /tmp. When a test suite has many
        #   tests, this can be a big disk cost (also, /tmp/ is frequently a
        #   RAM fs, which increases worries about capacity).
        # - don't have colliding filenames (hard to do without cluttering
        #   up /tmp)
        # - if we do have have colliding filenames, writes should at least
        #   avoid being racy.
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            python exception: {e}

            For Torch-MLIR developers, the error can be reproduced with:
            $ torch-mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise TorchMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


class OutputType(Enum):

    # Output torch dialect in backend form. When converting from TorchDynamo,
    # this comes after some decomposition and reduce op variants passes are
    # applied to the raw torch dialect. When converting from TorchScript, this
    # comes after some cleanup passes which attempt to de-alias, decompose and infer shapes.
    # These should be roughly the same level of abstraction since those
    # steps are done within PyTorch itself when coming directly from Dynamo/FX.
    TORCH = "torch"

    # The output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = "linalg-on-tensors"

    # This output type consists of `tosa` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to TOSA.
    TOSA = "tosa"

    # The output type contains a mix of `tosa`, `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (only standard mlir dialect ops). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `tosa` and `linalg`-on-tensors ops.
    # `torch` ops are first lowered to `tosa` as much as possible.
    # Remaining ops are then lowered to `linalg`-on-tensors.
    TOSA_LINALG = "tosa-linalg"

    # This output type consists of `stablehlo` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to StableHLO.
    STABLEHLO = "stablehlo"

    # Raw output of the JIT IR importer in the TorchScript frontend or that of
    # the FX IR importer in the TorchDynamo frontend. This is not expected to be useful
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
            raise ValueError(
                f"For output_type= argument, expected one of: "
                f"{', '.join(OutputType.__members__.keys())}"
            )
        return OutputType[spec]


def lower_mlir_module(verbose, output_type, module):
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    if output_type == OutputType.TORCH:
        return module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-tosa-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(module)
        return module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
        )
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(module)
        return module

    elif output_type == OutputType.TOSA_LINALG:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-tosa-linalg-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA + Linalg Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TOSA + Linalg Backend IR")
            print(module)
        return module

    elif output_type == OutputType.STABLEHLO:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO Backend IR",
        )
        if verbose:
            print("\n====================")
            print("StableHLO Backend IR")
            print(module)
        return module
    raise Exception(f"Unknown OutputType: {output_type}")
