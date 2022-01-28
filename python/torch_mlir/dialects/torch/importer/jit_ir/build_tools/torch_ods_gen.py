# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Queries the pytorch op registry and generates ODS and CC sources for the ops.
"""

from typing import Any, Dict, List, Optional, TextIO, Sequence, Tuple, Union

import argparse
from contextlib import contextmanager
import importlib
import io
import itertools
import logging
import os
import pprint
import re
import sys
import textwrap
import traceback

# Note that this utility exists only in the c-extension.
from torch_mlir._mlir_libs._jit_ir_importer import get_registered_ops # pytype: disable=import-error


class TextEmitter:
    """Helper for emitting text files"""
    _INDENT = "  "

    def __init__(self, out: TextIO):
        super().__init__()
        self.out = out
        self.indent_level = 0

    @contextmanager
    def indent(self, level: int = 1):
        self.indent_level += level
        yield
        self.indent_level -= level
        assert self.indent_level >= 0, "Unbalanced indentation"

    def print(self, s: str):
        current_indent = self._INDENT * self.indent_level
        for line in s.splitlines():
            self.out.write(current_indent + line + "\n")

    def quote(self, s: str) -> str:
        s = s.replace(r'"', r'\\"')
        return f'"{s}"'

    def quote_multiline_docstring(self, s: str, indent_level: int = 0) -> str:
        # TODO: Possibly find a python module to markdown the docstring for
        # better document generation.
        # Unlikely to contain the delimiter and since just a docstring, be safe.
        s = s.replace("}]", "")
        # Strip each line.
        s = "\n".join([l.rstrip() for l in s.splitlines()])
        indent = self._INDENT * indent_level
        s = textwrap.indent(s, indent + self._INDENT)
        return "[{\n" + s + "\n" + indent + "}]"


class JitOperator:
    """Information about a single registered `torch::jit::Operator`"""
    def __init__(self, op_info: "OP_INFO_DICT"):
        """Create a JitOperator from the raw OP_INFO_DICT extracted from
        the PyTorch JIT operator registry.
        """
        namespace, _, unqualified_name = op_info["name"][0].partition("::")
        self.namespace = namespace
        self.unqualified_name = unqualified_name
        self.overload_name = op_info["name"][1]
        self.is_c10_op = op_info["is_c10_op"]
        self.is_vararg = op_info["is_vararg"]
        self.is_varret = op_info["is_varret"]
        self.is_mutable = op_info["is_mutable"]
        self.arguments = op_info["arguments"]
        self.returns = op_info["returns"]

        self.unique_key = self.create_unique_key()

    def create_unique_key(self) -> str:
        """Create a unique, human-readable key for this JitOperator.

        The key consists of the operator name and its overload name, which
        together form a unique identifier. We also redundantly
        append a signature to the end, which gives some robustness to changes
        in PyTorch and also generally makes things more readable.
        The format is:
        ```
            namespace::unqualified_name[.overload_name] : (type1,type2) -> (type3,type4)
        ```
        This is a modified version of the signature strings seen throughout
        PyTorch. The main difference is the inclusion of return types and the
        extra spacing and `:`. E.g.the above would be just:
        ```
          namespace::kernel_name[.overload_name](type1,type2)
        ```
        (PyTorch doesn't canonically include the result types since they don't
        participate in their dispatch overload resolution, which is of primary
        concern for them)
        """
        overload = "" if not self.overload_name else f".{self.overload_name}"
        if self.is_vararg:
            arg_str = "..."
        else:
            arg_str = ", ".join(arg["type"] for arg in self.arguments)
        if self.is_varret:
            ret_str = "..."
        else:
            ret_str = ", ".join(ret["type"] for ret in self.returns)
        return f"{self.namespace}::{self.unqualified_name}{overload} : ({arg_str}) -> ({ret_str})"

    @property
    def triple(self):
        """Returns the unique 3-tuple identifying this operator.

        This is a useful alternative to the "unique name" for programmatic
        access, such as when needing to convert one op to a related op by
        a programmatic transformation of the triple.
        """
        return self.namespace, self.unqualified_name, self.overload_name

    def get_mlir_names(self):
        """Gets the MLIR op name (excluding `torch.`) and td def name.

        Not all ops are necessarily registered or in the .td file, but these
        are useful in the repr for cross referencing, and it's useful to have
        them in a single point of truth.
        """
        def uppercase_first_letter(s):
            if not s:
                return s
            return s[0].upper() + s[1:]

        op_name_atoms = [self.namespace, self.unqualified_name]
        if self.overload_name:
            op_name_atoms.append(self.overload_name)
        op_name = ".".join(op_name_atoms)

        op_class_name_atoms = []
        for op_name_atom in op_name_atoms:
            for s in op_name_atom.split("_"):
                op_class_name_atoms.append(s if s else "_")
        td_def_name = "Torch_" + "".join(
            uppercase_first_letter(s) for s in op_class_name_atoms) + "Op"
        return op_name, td_def_name

    def __repr__(self):
        f = io.StringIO()
        emitter = TextEmitter(f)
        p = lambda *args: emitter.print(*args)
        p(f"JitOperator '{self.unique_key}':")
        with emitter.indent():

            # Emit the MLIR names to allow easy reverse lookup if starting
            # from an unregistered op.
            op_name, td_def_name = self.get_mlir_names()
            p(f"MLIR op name = torch.{op_name}")
            p(f"MLIR td def name = {td_def_name}")

            p(f"namespace = {self.namespace}")
            p(f"unqualified_name = {self.unqualified_name}")
            p(f"overload_name = {self.overload_name}")
            p(f"is_c10_op = {self.is_c10_op}")
            p(f"is_vararg = {self.is_vararg}")
            p(f"is_varret = {self.is_varret}")
            p(f"is_mutable = {self.is_mutable}")
            if not self.arguments:
                p("arguments = []")
            else:
                p("arguments:")
                with emitter.indent():
                    for arg in self.arguments:
                        p(f"arg: {arg}")
            if not self.returns:
                p("returns = []")
            else:
                p("returns:")
                with emitter.indent():
                    for ret in self.returns:
                        p(f"ret: {ret}")
        return f.getvalue()


class Registry:
    """An indexed collection of JitOperators"""
    def __init__(self, operators: List[JitOperator]):
        self.by_unique_key = {}
        self.by_triple = {}
        for o in operators:
            self.by_unique_key[o.unique_key] = o
            self.by_triple[o.triple] = o

    def __getitem__(self, key: str):
        """Looks up a JitOperator by its "unique key"."""
        return self.by_unique_key[key]

    def get_by_triple(self, key: Tuple[str, str, str]):
        """Looks up a JitOperator by its unique "triple"."""
        return self.by_triple[key]


# A List[Dict[str, _]] mapping attribute names to:
#   - str (e.g. {'name': 'dim'} )
#   - int (e.g. {'N': 1} )
#   - Dict[str, List[str]]
#       (e.g. {'alias_info': {'before': ['alias::a'], 'after': ['alias::a']}} )
SIGLIST_TYPE = List[Dict[str, Union[str, int, Dict[str, List[str]]]]]
# A Dict[str, _] describing a registered op. Each field is either
#   - bool (e.g. {'is_mutable': False} )
#   - Tuple[str] (e.g. {'name': ('aten::size', 'int')} )
#   - SIGLIST_TYPE (e.g. {'arguments': [...], 'returns': [...]} )
OP_INFO_DICT = Dict[str, Union[bool, Tuple[str], SIGLIST_TYPE]]

# Mapping from torch types to their corresponding ODS type predicates.
# Use `get_ods_type` instead of using this directly.
TORCH_TYPE_TO_ODS_TYPE = {
    "Tensor": "AnyTorchTensorType",
    "Tensor?": "AnyTorchOptionalTensorType",
    "Tensor?[]": "AnyTorchOptionalTensorListType",
    "Tensor[]": "AnyTorchTensorListType",
    "Scalar": "AnyTorchScalarType",
    "Scalar?": "AnyTorchOptionalScalarType",
    "int": "Torch_IntType",
    "int[]": "TorchIntListType",
    "int?": "TorchOptionalIntType",
    "bool": "Torch_BoolType",
    "bool[]": "TorchBoolListType",
    "bool?": "TorchOptionalBoolType",
    "float": "Torch_FloatType",
    "t[]": "AnyTorchListType",
    "t": "AnyTorchType",
    "t1": "AnyTorchType",
    "t2": "AnyTorchType",
    "Any": "AnyTorchType",
    "Device": "Torch_DeviceType",
    "Device?": "TorchOptionalDeviceType",
    "Generator": "Torch_GeneratorType",
    "Generator?": "TorchOptionalGeneratorType",
    "str": "Torch_StringType",
    "str?": "TorchOptionalStringType",
    "str[]": "TorchStringListType",
    "Dict": "Torch_DictType",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase": "Torch_LinearParamsType",
}


def get_ods_type(type: str):
    # TODO: Increase precision on dict type modeling.
    if type.startswith("Dict("):
      type = "Dict"
    ods_type = TORCH_TYPE_TO_ODS_TYPE.get(type)
    if ods_type is None:
        raise Exception(
            f"{type!r} not in TORCH_TYPE_TO_ODS_TYPE mapping. Please add it!")
    return ods_type


def _get_main_module_name() -> str:
    # pytype: disable=attribute-error
    return sys.modules["__main__"].__loader__.name
    # pytype: enable=attribute-error


ODS_BANNER = "\n".join([
    "//===-------------------------------------------------------*- tablegen -*-===//",
    "//",
    "// This file is licensed under the Apache License v2.0 with LLVM Exceptions.",
    "// See https://llvm.org/LICENSE.txt for license information.",
    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
    "// Also available under a BSD-style license. See LICENSE.",
    "//",
    "// Operation summaries and descriptions were systematically derived from public",
    "// API docstrings and are licensed accordingly:",
    "//   https://github.com/pytorch/pytorch/blob/master/LICENSE",
    "//===----------------------------------------------------------------------===//",
    "//",
    "// This file is automatically generated.  Please do not edit.",
    "// Generated via:",
    f"//   python -m {_get_main_module_name()}",
    "//",
    "//===----------------------------------------------------------------------===//",
    "",
    "",
])


def raw_emit_op(operator: JitOperator, f: TextIO, *, traits: List[str],
                has_folder: bool, has_canonicalizer: bool):
    """Emit the ODS for a JitOperator to a textual file.

    This is the lowest level of emission and is responsible for low-level
    textual emission details. This function should not have any "smarts"
    for deducing traits/etc.

    You probably don't want to call this directly.
    """
    emitter = TextEmitter(f)
    p = lambda *args: emitter.print(*args)
    op_name, td_def_name = operator.get_mlir_names()

    # Generate unique result names for ops with nameless results
    multiple_results = len(operator.returns) > 1
    generic_result_name = lambda i: "result" + (str(i) if multiple_results else "")

    p(f"def {td_def_name} : Torch_Op<{emitter.quote(op_name)}, [")
    with emitter.indent():
        with emitter.indent():
            p(",\n".join(traits))
        p("]> {")
    with emitter.indent():
        summary = f"Generated op for `{operator.unique_key}`"
        p(f"let summary = {emitter.quote(summary)};")
        p(f"let arguments = (ins")
        with emitter.indent():
            if operator.is_vararg:
                p("Variadic<AnyTorchType>:$operands")
            else:
                p(",\n".join([
                    f"""{get_ods_type(arg["type"])}:${arg["name"]}"""
                    for arg in operator.arguments
                ]))
        p(");")
        p(f"let results = (outs")
        with emitter.indent():
            if operator.is_varret:
                p("Variadic<AnyTorchType>:$results")
            else:
                p(",\n".join([
                    f"""{get_ods_type(ret["type"])}:${ret["name"] or generic_result_name(e)}"""
                    for e, ret in enumerate(operator.returns)
                ]))
        p(");")

        if operator.is_vararg:
            assembly_operands = "`(` $operands `)`"
            assembly_operand_types = "qualified(type($operands))"
        else:
            assembly_operands = " `,` ".join("$" + arg["name"]
                                             for arg in operator.arguments)
            assembly_operand_types = " `,` ".join(
                f"""qualified(type(${arg["name"]}))""" for arg in operator.arguments)
        if operator.is_varret:
            assembly_result_types = "qualified(type($results))"
        else:
            assembly_result_types = " `,` ".join(
                f"""qualified(type(${ret["name"] or generic_result_name(e)}))"""
                for e, ret in enumerate(operator.returns))
        if assembly_operand_types and assembly_result_types:
            maybe_arrow = " `->` "
        else:
            maybe_arrow = ""
        assembly_format = f"{assembly_operands} attr-dict `:` {assembly_operand_types}{maybe_arrow}{assembly_result_types}"
        p(f"let assemblyFormat = {emitter.quote(assembly_format)};")
        if has_folder:
            p("let hasFolder = 1;")
        if has_canonicalizer:
            p("let hasCanonicalizer = 1;")
    p("}")
    p("\n")


def emit_op(operator: JitOperator,
            f: TextIO,
            *,
            traits: Optional[List[str]] = None,
            has_folder: bool = False,
            has_canonicalizer: bool = False):
    """Main entry point for op emission.

    Besides emitting the op, it deduces / adds traits based on the operator
    information.
    """
    if traits is None:
        traits = []

    # All Torch operators allow type refinement.
    traits += ["AllowsTypeRefinement"]
    # If no operands have aliasing relations, then the op has value semantics.
    # Note that this is different from MLIR's NoSideEffect which is much
    # stronger (for example, it cannot be applied to ops that might emit errors
    # when operand shapes mismatch).
    if not operator.is_vararg and not operator.is_varret and all(
            "alias_info" not in x
            for x in itertools.chain(operator.arguments, operator.returns)):
      # It seems the FunctionSchema of "prim::unchecked_cast : (t) -> (t)" has
      # incorrect alias information. The result can alias with other tensors
      # but the alias annotation is empty.
      if operator.unique_key != "prim::unchecked_cast : (t) -> (t)":
          traits += ["HasValueSemantics"]

    raw_emit_op(operator,
                f,
                traits=traits,
                has_folder=has_folder,
                has_canonicalizer=has_canonicalizer)


def emit_prim_ops(torch_ir_dir: str, registry: Registry):
    td_file = os.path.join(torch_ir_dir, "GeneratedPrimOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        emit("prim::layout : (Tensor) -> (int)")
        emit("prim::TupleIndex : (Any, int) -> (Any)", has_canonicalizer=True)
        emit("prim::device : (Tensor) -> (Device)")
        emit("prim::dtype : (Tensor) -> (int)", has_folder=True)
        emit("prim::TupleUnpack : (Any) -> (...)", has_canonicalizer=True)
        emit("prim::NumToTensor.Scalar : (Scalar) -> (Tensor)")
        emit("prim::min.self_int : (int[]) -> (int)")
        emit("prim::min.int : (int, int) -> (int)")
        emit("prim::max.self_int : (int[]) -> (int)")
        emit("prim::max.int : (int, int) -> (int)")
        emit("prim::RaiseException : (str, str?) -> ()")
        emit("prim::Uninitialized : () -> (Any)", traits=["NoSideEffect"])
        emit("prim::unchecked_cast : (t) -> (t)", has_folder=True,
             traits=["DeclareOpInterfaceMethods<CastOpInterface>"])
        emit("prim::Print : (...) -> ()")
        emit("prim::tolist : (...) -> (...)")


def emit_aten_ops(torch_ir_dir: str, registry: Registry):
    # Note the deliberate lowercasing of the "t" for consistency with all
    # the name munging. This is not load bearing, but is convenient for
    # consistency.
    td_file = os.path.join(torch_ir_dir, "GeneratedAtenOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        def emit_with_mutating_variants(key, **kwargs):
            operator = registry[key]
            emit_op(operator, f, **kwargs)
            ns, unqual, overload = operator.triple
            emit_op(registry.get_by_triple((ns, unqual + "_", overload)),
                    f,
                    traits=["IsTrailingUnderscoreInplaceVariant"])

        # Elementwise tensor compute ops
        for key in [
                "aten::tanh : (Tensor) -> (Tensor)",
                "aten::relu : (Tensor) -> (Tensor)",
                "aten::leaky_relu : (Tensor, Scalar) -> (Tensor)",
                "aten::log : (Tensor) -> (Tensor)",
                "aten::sigmoid : (Tensor) -> (Tensor)",
                "aten::sin : (Tensor) -> (Tensor)",
                "aten::exp : (Tensor) -> (Tensor)",
                "aten::cos : (Tensor) -> (Tensor)",
                "aten::neg : (Tensor) -> (Tensor)",
                "aten::floor : (Tensor) -> (Tensor)",
                "aten::ceil : (Tensor) -> (Tensor)",
                "aten::bitwise_not : (Tensor) -> (Tensor)",
                "aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::div.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)",
                "aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)",
                "aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)",
                "aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::div.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)",
                "aten::log2 : (Tensor) -> (Tensor)",
                "aten::rsqrt : (Tensor) -> (Tensor)",
                "aten::abs : (Tensor) -> (Tensor)",
                "aten::reciprocal : (Tensor) -> (Tensor)",
                "aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)",
                "aten::square : (Tensor) -> (Tensor)",

        ]:
            emit_with_mutating_variants(key)
        # Elementwise tensor compute ops that don't have the standard mutating
        # variants.
        emit("aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)")
        emit("aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)")
        emit("aten::maximum : (Tensor, Tensor) -> (Tensor)")
        emit("aten::minimum : (Tensor, Tensor) -> (Tensor)")
        emit("aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)")
        emit("aten::gelu : (Tensor) -> (Tensor)")
        emit("aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)")

        # Ops without value semantics but the corresponding without trailing
        # underscore variant doesn't exist.
        emit("aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)")
        emit("aten::bernoulli : (Tensor, Generator?) -> (Tensor)")
        emit("aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)")

        emit_with_mutating_variants("aten::triu : (Tensor, int) -> (Tensor)")
        emit_with_mutating_variants("aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)")

        # Non-elementwise tensor compute ops
        emit("aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)")
        emit("aten::mm : (Tensor, Tensor) -> (Tensor)")
        emit("aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)")
        emit("aten::matmul : (Tensor, Tensor) -> (Tensor)")
        emit(
            "aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)"
        )
        emit(
            "aten::native_batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float) -> (Tensor, Tensor, Tensor)"
        )
        emit(
            "aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)"
        )
        emit(
            "aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)"
        )
        emit (
            "aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)"
        )
        emit(
            "aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)"
        )
        emit(
            "aten::softmax.int : (Tensor, int, int?) -> (Tensor)"
        )
        emit(
            "aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)"
        )
        emit(
            "aten::_log_softmax : (Tensor, int, bool) -> (Tensor)"
        )
        emit("aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)")
        emit("aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)")
        emit("aten::transpose.int : (Tensor, int, int) -> (Tensor)")
        emit("aten::permute : (Tensor, int[]) -> (Tensor)")
        emit("aten::bmm : (Tensor, Tensor) -> (Tensor)")
        emit("aten::cumsum : (Tensor, int, int?) -> (Tensor)")
        emit("aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::logsumexp : (Tensor, int[], bool) -> (Tensor)")
        emit("aten::mean.dim : (Tensor, int[], bool, int?) -> (Tensor)")
        emit("aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)")
        emit("aten::sqrt : (Tensor) -> (Tensor)")
        emit("aten::_softmax : (Tensor, int, bool) -> (Tensor)")
        emit("aten::mean : (Tensor, int?) -> (Tensor)")
        emit("aten::std : (Tensor, bool) -> (Tensor)")
        emit("aten::var : (Tensor, bool) -> (Tensor)")
        emit("aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)")
        emit("aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)")

        # Misc tensor ops.
        emit("aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)")
        emit("aten::squeeze.dim : (Tensor, int) -> (Tensor)", has_folder=True)
        emit("aten::unsqueeze : (Tensor, int) -> (Tensor)")
        emit("aten::squeeze : (Tensor) -> (Tensor)", has_folder=True)
        emit("aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)")
        emit("aten::dim : (Tensor) -> (int)", has_folder=True)
        emit("aten::size : (Tensor) -> (int[])", has_canonicalizer=True)
        emit("aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::Bool.Tensor : (Tensor) -> (bool)")
        emit("aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::tensor : (t[], int?, Device?, bool) -> (Tensor)")
        emit("aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)")
        emit("aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)")
        emit("aten::_shape_as_tensor : (Tensor) -> (Tensor)")
        emit("aten::all : (Tensor) -> (Tensor)")
        emit("aten::any : (Tensor) -> (Tensor)")
        emit("aten::any.dim : (Tensor, int, bool) -> (Tensor)")
        emit("aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::argmax : (Tensor, int?, bool) -> (Tensor)")
        emit("aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)")
        emit("aten::clone : (Tensor, int?) -> (Tensor)")
        emit("aten::contiguous : (Tensor, int) -> (Tensor)")
        emit("aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)")
        emit("aten::detach : (Tensor) -> (Tensor)")
        emit("aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)")
        emit("aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::expand : (Tensor, int[], bool) -> (Tensor)")
        emit("aten::broadcast_to : (Tensor, int[]) -> (Tensor)")
        emit("aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)")
        emit("aten::index_select : (Tensor, int, Tensor) -> (Tensor)")
        emit("aten::item : (Tensor) -> (Scalar)")
        emit("aten::masked_select : (Tensor, Tensor) -> (Tensor)")
        emit("aten::numel : (Tensor) -> (int)")
        emit("aten::repeat : (Tensor, int[]) -> (Tensor)")
        emit("aten::reshape : (Tensor, int[]) -> (Tensor)")
        emit("aten::resize_ : (Tensor, int[], int?) -> (Tensor)")
        emit("aten::select.int : (Tensor, int, int) -> (Tensor)")
        emit("aten::size.int : (Tensor, int) -> (int)", has_folder=True)
        emit("aten::stack : (Tensor[], int) -> (Tensor)")
        emit("aten::sum : (Tensor, int?) -> (Tensor)")
        emit("aten::sum.dim_IntList : (Tensor, int[], bool, int?) -> (Tensor)")
        emit("aten::max : (Tensor) -> (Tensor)")
        emit("aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)")
        emit("aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)", has_folder=True)
        emit("aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)")
        emit("aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)")
        emit("aten::type_as : (Tensor, Tensor) -> (Tensor)")
        emit("aten::view : (Tensor, int[]) -> (Tensor)", has_folder=True)
        emit("aten::_unsafe_view : (Tensor, int[]) -> (Tensor)")
        emit("aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)")
        emit("aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)")
        emit("aten::len.Tensor : (Tensor) -> (int)")
        emit("aten::cpu : (Tensor) -> (Tensor)")
        emit("aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)")
        emit("aten::IntImplicit : (Tensor) -> (int)")
        emit("aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)")
        emit("aten::Int.Tensor : (Tensor) -> (int)", has_folder=True)
        emit("aten::dropout : (Tensor, float, bool) -> (Tensor)")
        emit("aten::t : (Tensor) -> (Tensor)")

        # Dict ops.
        emit("aten::__contains__.str : (Dict(str, t), str) -> (bool)", has_folder=True)
        emit("aten::__getitem__.Dict_str : (Dict(str, t), str) -> (t)", has_folder=True)
        emit("aten::_set_item.str : (Dict(str, t), str, t) -> ()")
        emit("aten::keys.str : (Dict(str, t)) -> (str[])")
        emit("aten::get.default_str : (Dict(str, t), str, t) -> (t)")
        emit("aten::Delete.Dict_str : (Dict(str, t), str) -> ()")

        # List ops.
        emit("aten::cat : (Tensor[], int) -> (Tensor)")
        emit("aten::append.t : (t[], t) -> (t[])")
        emit("aten::add.t : (t[], t[]) -> (t[])")
        emit("aten::eq.int_list : (int[], int[]) -> (bool)")
        emit("aten::list.t : (t[]) -> (t[])")
        emit("aten::slice.t : (t[], int?, int?, int) -> (t[])")

        # Str ops.
        emit("aten::add.str : (str, str) -> (str)")
        emit("aten::eq.str : (str, str) -> (bool)", has_folder=True)
        emit("aten::str : (t) -> (str)")
        emit("aten::format : (...) -> (str)")
        emit("aten::join : (str, str[]) -> (str)")

        # Type conversion ops.
        emit("aten::Float.Scalar : (Scalar) -> (float)")
        emit("aten::Float.str : (str) -> (float)")
        emit("aten::Int.float : (float) -> (int)")

        # Primitive ops
        emit("aten::gt.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::ge.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::lt.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::le.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::ne.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::eq.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::floordiv.int : (int, int) -> (int)", has_folder=True)
        emit("aten::remainder.int : (int, int) -> (int)", has_folder=True)
        emit("aten::add.int : (int, int) -> (int)", has_folder=True)
        emit("aten::sub.int : (int, int) -> (int)", has_folder=True)
        emit("aten::mul.int : (int, int) -> (int)", has_folder=True)
        emit("aten::neg.int : (int) -> (int)", has_folder=True)
        emit("aten::log.int : (int) -> (float)")
        emit("aten::add.float_int : (float, int) -> (float)")
        emit("aten::mul.float : (float, float) -> (float)")
        emit("aten::neg.float : (float) -> (float)")
        emit("aten::lt.float_int : (float, int) -> (bool)")
        emit("aten::eq.float : (float, float) -> (bool)", has_folder=True)
        emit("aten::__and__.bool : (bool, bool) -> (bool)")
        emit("aten::ne.bool : (bool, bool) -> (bool)", has_folder=True)
        emit("aten::__is__ : (t1, t2) -> (bool)", has_folder=True)
        emit("aten::__isnot__ : (t1, t2) -> (bool)", has_folder=True)
        emit("aten::__not__ : (bool) -> (bool)", has_folder=True)
        emit("aten::len.t : (t[]) -> (int)",
             has_folder=True,
             has_canonicalizer=True)
        emit("aten::__getitem__.t : (t[], int) -> (t)", has_canonicalizer=True)
        emit("aten::_set_item.t : (t[], int, t) -> (t[])")
        emit("aten::div : (Scalar, Scalar) -> (float)")
        emit("aten::eq.device : (Device, Device) -> (bool)")

        # backprop ops
        emit("aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)")
        emit("aten::tanh_backward : (Tensor, Tensor) -> (Tensor)")
        emit("aten::gelu_backward : (Tensor, Tensor) -> (Tensor)")
        emit("aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)")
        emit("aten::native_layer_norm_backward : (Tensor, Tensor, int[], Tensor, Tensor, Tensor?, Tensor?, bool[]) -> (Tensor, Tensor, Tensor)")



def emit_quantized_ops(torch_ir_dir: str, registry: Registry):
    td_file = os.path.join(torch_ir_dir, "GeneratedQuantizedOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        emit(
            "quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)",
            traits=["HasValueSemantics"])


def dump_registered_ops(outfile: TextIO, registry: Registry):
    for _, v in sorted(registry.by_unique_key.items()):
        outfile.write(repr(v))


def load_registry() -> Registry:
    return Registry([JitOperator(op_info) for op_info in get_registered_ops()])


def main(args: argparse.Namespace):
    registry = load_registry()
    if args.debug_registry_dump:
        with open(args.debug_registry_dump, "w") as debug_registry_dump:
            dump_registered_ops(debug_registry_dump, registry)
    emit_prim_ops(args.torch_ir_dir, registry)
    emit_aten_ops(args.torch_ir_dir, registry)
    emit_quantized_ops(args.torch_ir_dir, registry)


def _create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_ods")
    parser.add_argument(
        "--torch_ir_dir",
        required=True,
        help="Directory containing the Torch dialect definition")
    parser.add_argument(
        "--debug_registry_dump",
        help="File to dump the the PyTorch JIT operator registry into")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = _create_argparse()
    args = parser.parse_args()
    main(args)
