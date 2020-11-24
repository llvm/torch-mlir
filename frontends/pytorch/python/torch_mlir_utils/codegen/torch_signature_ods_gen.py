#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Queries the pytorch op registry and generates ODS and CC sources for the ops.
"""

from typing import Any, Dict, List, Optional, TextIO, Sequence, Tuple, Union

import argparse
from contextlib import contextmanager
import importlib
import logging
import re
import sys
import textwrap
import traceback

# Note that this utility exists only in the c-extension.
from _torch_mlir import get_registered_ops  # pytype: disable=import-error

# A Dist[str, _] mapping 'aten::OpName' to:
#   - bool (e.g. {'is_mutable': False} )
#   - Tuple[str] (e.g. {'name': ('aten::size', 'int')} )
#   - SIGLIST_TYPE (e.g. {'arguments': [...], 'returns': [...]} )
REG_OP_TYPE = Dict[str, Union[bool, Tuple[str], "SIGLIST_TYPE"]]
# A List[Dict[str, _]] mapping attribute names to:
#   - str (e.g. {'name': 'dim'} )
#   - int (e.g. {'N': 1} )
#   - Dict[str, List[str]]
#       (e.g. {'alias_info': {'before': ['alias::a'], 'after': ['alias::a']}} )
SIGLIST_TYPE = List[Dict[str, Union[str, int, Dict[str, List[str]]]]]


def _create_argparse() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="generate_ods")
  parser.add_argument("--ods_td_file",
                      required=True,
                      help="File to write the generated ODS to")
  parser.add_argument("--ods_impl_file",
                      required=True,
                      help="CC include file to include in ops implementation")
  parser.add_argument("--debug_op_reg_file",
                      help="Write out a file of op registrations")
  return parser


def main(args: argparse.Namespace):
  reg_ops = _load_ops_as_dict()
  if args.debug_op_reg_file:
    with open(args.debug_op_reg_file, "w") as debug_ops_file:
      dump_registered_ops(debug_ops_file, reg_ops)

  with open(args.ods_td_file, "w") as ods_file, open(args.ods_impl_file,
                                                     "w") as impl_file:
    ods_emitter = OdsEmitter(ods_file)
    ods_emitter.print(ODS_BANNER)
    impl_emitter = CCImplEmitter(impl_file)
    impl_emitter.print(CC_IMPL_BANNER)
    generator = OpGenerator(reg_ops, ods_emitter, impl_emitter)
    generate_ops(generator)


def generate_ops(g: "OpGenerator"):
  # "Binary"-ops.
  # There are some variation in these, so we spell them out in case if they
  # need individual customization.
  g.print_banner("Binary arithmetic ops")
  g.ordinary_binary_op("aten::add(Tensor,Tensor,Scalar)", "AddOp", "add")
  g.ordinary_binary_op("aten::atan2(Tensor,Tensor)", "Atan2Op", "atan2")
  g.ordinary_binary_op("aten::div(Tensor,Tensor)", "DivOp", "div")
  g.ordinary_binary_op("aten::floor_divide(Tensor,Tensor)", "FloorDivideOp",
                       "floor_divide")
  g.ordinary_binary_op("aten::mul(Tensor,Tensor)", "MulOp", "mul")
  g.ordinary_binary_op("aten::remainder(Tensor,Tensor)", "RemainderOp",
                       "remainder")
  g.ordinary_binary_op("aten::true_divide(Tensor,Tensor)", "TrueDivideOp",
                       "true_divide")

  g.ordinary_binary_op("aten::maximum(Tensor,Tensor)", "MaximumOp", "maximum")
  g.ordinary_binary_op("aten::minimum(Tensor,Tensor)", "MinimumOp", "minimum")

  # Unary-ops. These are all the same so just name munge them.
  g.print_banner("Unary arithmetic ops")
  for uname in [
      "abs", "acos", "angle", "asin", "atan", "ceil", "conj", "cos", "cosh",
      "digamma", "erf", "erfc", "erfinv", "exp", "expm1", "floor", "frac",
      "lgamma", "log", "log10", "log1p", "log2", "neg", "relu", "reciprocal",
      "round", "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqrt", "tan", "tanh",
      "trunc"
  ]:
    g.ordinary_unary_op(f"aten::{uname}(Tensor)",
                        f"{snakecase_to_camelcase(uname)}Op", uname)

  # Convolution ops. Note that these are special in PyTorch and the importer,
  # and we model them after the signatures of the convolution_overrideable
  # ops (generic for non-CPU/GPU backends) but set the names according to
  # how they come in.
  g.print_banner("NN ops")
  g.ordinary_immutable_op(
      "aten::convolution_overrideable(Tensor,Tensor,Tensor?,int[],int[],int[],bool,int[],int)",
      "ConvolutionOp",
      "convolution",
      alias_kernel_names=["aten::convolution"])
  g.ordinary_immutable_op(
      "aten::convolution_backward_overrideable(Tensor,Tensor,Tensor,int[],int[],int[],bool,int[],int,bool[])",
      "ConvolutionBackwardOp",
      "convolution_backward",
      alias_kernel_names=["aten::convolution_backward"],
      # These do return as None but are not coded optional in the registry :(
      override_return_types=["Tensor?", "Tensor?", "Tensor?"])

  g.ordinary_immutable_op("aten::_log_softmax(Tensor,int,bool)", "LogSoftmaxOp",
                          "log_softmax")
  g.ordinary_immutable_op(
      "aten::_log_softmax_backward_data(Tensor,Tensor,int,Tensor)",
      "LogSoftmaxBackwardDataOp", "log_softmax_backward_data")
  g.ordinary_immutable_op("aten::mm(Tensor,Tensor)", "MmOp", "mm")

  # Loss functions.
  g.print_banner("Loss function ops")
  g.ordinary_immutable_op(
      "aten::nll_loss_forward(Tensor,Tensor,Tensor?,int,int)",
      "NllLossForwardOp", "nll_loss_forward")
  # Note also a grad_input 8-arg variant.
  g.ordinary_immutable_op(
      "aten::nll_loss_backward(Tensor,Tensor,Tensor,Tensor?,int,int,Tensor)",
      "NllLossBackwardOp", "nll_loss_backward")

  g.ordinary_immutable_op(
      "aten::nll_loss2d_forward(Tensor,Tensor,Tensor?,int,int)",
      "NllLoss2dForwardOp", "nll_loss2d_forward")
  # Note also a grad_input 8-arg variant.
  g.ordinary_immutable_op(
      "aten::nll_loss2d_backward(Tensor,Tensor,Tensor,Tensor?,int,int,Tensor)",
      "NllLoss2dBackwardOp", "nll_loss2d_backward")

  # One-off in-place ops (note that many in-place arithmetic ops are handled
  # as a transformation from their immutable forms).
  g.ordinary_inplace_op("aten::copy_(Tensor,Tensor,bool)",
                        "CopyInplaceOp",
                        "copy.inplace",
                        drop_arg_indices=[2])


def dump_registered_ops(outfile: TextIO, reg_ops_dict: Dict[str, REG_OP_TYPE]):
  for k in sorted(reg_ops_dict.keys()):
    attr_dict = reg_ops_dict[k]
    outfile.write(f"OP '{k}':\n")
    for attr_name, attr_value in attr_dict.items():
      outfile.write(f"    {attr_name} = {attr_value!r}\n")
    outfile.write("\n")


class OpGenerator:

  def __init__(self, reg_ops: Dict[str, REG_OP_TYPE], ods_emitter: "OdsEmitter",
               impl_emitter: "CCImplEmitter"):
    super().__init__()
    self.reg_ops = reg_ops
    self.ods_emitter = ods_emitter
    self.impl_emitter = impl_emitter

  def print_banner(self, text: str):
    seperator = f"// {'-' * 77}"
    for em in (self.ods_emitter, self.impl_emitter):
      em.print(seperator)
      em.print(f"// {text}")
      em.print(seperator)
      em.print("")

  def define_op(self, kernel_sig: str, ods_name: str, op_name: str,
                **kwargs) -> "InflightOpDef":
    return InflightOpDef(self,
                         kernel_sig=kernel_sig,
                         ods_name=ods_name,
                         op_name=op_name,
                         **kwargs)

  def ordinary_binary_op(self,
                         kernel_sig: str,
                         ods_name: str,
                         op_name: str,
                         promote_trailing_out_tensor: bool = True,
                         traits: Sequence[str] = (),
                         **kwargs):
    """"Binary"-ops. These ops typically have:
      - '.Tensor' variant where the second arg is a Tensor
      - '.Scalar' variant where the second arg is a Scalar
      - An '.out' variant which contains a final "outref" argument
    Actual suffixes vary and the rules are set up to match anything
    that comes in in one of these forms.
    In addition, most of these have in-place versions, possibly of the
    '.Tensor' and '.Scalar' variants above.
    Note that many of these have more than two arguments (i.e. alpha/beta
    scalars trailing and such), but they are not relevant for
    matching/conversions (if they are more than pass-through, then have a
    dedicated rule).
    We generally canonicalize all of these forms to a single recognized op
    by:
      - Enabling the flag to promotTrailingOutTensor
      - Enabling the flag matchInplaceVariant
      - Setting all arguments and returns to kImmutableTensor
      - Enabling kPromoteScalarToTensor on the second argument.
    """
    opdef = self.define_op(
        kernel_sig=kernel_sig,
        ods_name=ods_name,
        op_name=op_name,
        promote_trailing_out_tensor=promote_trailing_out_tensor,
        traits=list(traits) + ["NoSideEffect"],
        **kwargs)
    opdef.arg_transforms(
        type_transforms={
            "Tensor:0": "AnyTorchImmutableTensor",
            "Tensor:1": "AnyTorchImmutableTensor",
            "Scalar:1": "AnyTorchImmutableTensor",
            "Scalar": "AnyTorchScalarType",
        },
        flag_transforms={
            ":0": ["kImmutableTensor"],
            ":1": ["kImmutableTensor", "kPromoteScalar"],
        },
    )
    opdef.return_transforms(
        type_transforms={
            "Tensor:0": "AnyTorchImmutableTensor",
        },
        flag_transforms={
            ":0": ["kImmutableTensor"],
        },
    )
    opdef.emit()

  def ordinary_immutable_op(self,
                            kernel_sig: str,
                            ods_name: str,
                            op_name: str,
                            promote_trailing_out_tensor: bool = True,
                            traits: Sequence[str] = (),
                            **kwargs):
    """"An ordinary immutable-tensor based op."""
    opdef = self.define_op(
        kernel_sig=kernel_sig,
        ods_name=ods_name,
        op_name=op_name,
        promote_trailing_out_tensor=promote_trailing_out_tensor,
        traits=list(traits) + ["NoSideEffect"],
        **kwargs)
    opdef.transforms(
        type_transforms={
            "Tensor": "AnyTorchImmutableTensor",
            "Tensor?": "AnyTorchOptionalImmutableTensor",
            "int": "AnyTorchIntType",
            "int[]": "AnyTorchIntListType",
            "bool": "AnyTorchBoolType",
            "bool[]": "AnyTorchBoolListType",
        },
        flag_transforms={
            "Tensor": ["kImmutableTensor"],
            "Tensor?": ["kImmutableTensor"],
        },
    )
    opdef.emit()

  def ordinary_inplace_op(self, kernel_sig: str, ods_name: str, op_name: str,
                          **kwargs):
    """In-place ops (ending in '_').

    These ops take a mutable first argument and then standard immutable
    conversions for subsequent. When emitting into MLIR, the return value is
    dropped.
    """
    opdef = self.define_op(kernel_sig=kernel_sig,
                           ods_name=ods_name,
                           op_name=op_name,
                           **kwargs)
    opdef.arg_transforms(
        type_transforms={
            ":0": "AnyTorchMutableTensor",
            "Tensor": "AnyTorchImmutableTensor",
            "Tensor?": "AnyTorchOptionalImmutableTensor",
            "int": "AnyTorchIntType",
            "int[]": "AnyTorchIntListType",
            "bool": "AnyTorchBoolType",
            "bool[]": "AnyTorchBoolListType",
        },
        flag_transforms={
            ":0": [],
            "Tensor": ["kImmutableTensor"],
            "Tensor?": ["kImmutableTensor"],
        },
    )
    opdef.return_transforms(
        type_transforms={
            ":0": "DROP_UNUSED",  # Ignored because we clear the outs below.
        },
        flag_transforms={
            ":0": ["kDropReturnAndAliasArg0"],
        },
    )
    opdef.map_signatures()
    opdef.ods_outs = []  # Clear the computed outs.
    opdef.emit()

  def ordinary_unary_op(self,
                        kernel_sig: str,
                        ods_name: str,
                        op_name: str,
                        promote_trailing_out_tensor: bool = True,
                        traits: Sequence[str] = (),
                        **kwargs):
    """Unary ops.

    These take and return a tensor and typically have an out and inplace
    variant (they may not but we generate patterns to match anyway).
    """
    opdef = self.define_op(
        kernel_sig=kernel_sig,
        ods_name=ods_name,
        op_name=op_name,
        promote_trailing_out_tensor=promote_trailing_out_tensor,
        traits=list(traits) + ["NoSideEffect"],
        **kwargs)
    opdef.arg_transforms(
        type_transforms={
            "Tensor:0": "AnyTorchImmutableTensor",
        },
        flag_transforms={
            ":0": ["kImmutableTensor"],
        },
    )
    opdef.return_transforms(
        type_transforms={
            "Tensor:0": "AnyTorchImmutableTensor",
        },
        flag_transforms={
            ":0": ["kImmutableTensor"],
        },
    )
    opdef.emit()

  def get_reg_record(self, kernel_sig: str) -> REG_OP_TYPE:
    """Gets the op-dict for a given registered op name.

    Args:
      kernel_sig: Signature of the kernel to find.
    Returns:
      Dict of the registration record.
    """
    record = self.reg_ops.get(kernel_sig)
    if record:
      return record

    # Try to give a nice "did you mean" style error, since this happens
    # so much.
    kernel_name, *rest = kernel_sig.split("(", maxsplit=1)
    dym_list = [k for k in self.reg_ops.keys() if k.startswith(kernel_name)]
    dym_message = '\n  '.join(dym_list)
    raise ValueError(f"Could not find registry op matching '{kernel_sig}'. "
                     f"Possible matches:\n  {dym_message}")

  def _map_sigtypes(
      self,
      siglist: SIGLIST_TYPE,
      type_transforms: Dict[str, str],
      flag_transforms: Dict[str, List[str]],
      drop_indices: Sequence[int] = (),
      override_types: Optional[Sequence[str]] = None,
  ) -> List[Tuple[str]]:
    """Maps a list of signature entries to ods dags and flag lists.

    The torch signature list contains dicts that minimally have keys 'name' and
    'type', representing torch names and types. Returns a corresponding
    list of 2-tuples of (ods_name, ods_type).

    The provided type_transforms is a dict of type substitutions, one of which
    must match for each entry in the list. The keys can be either a verbatim
    torch type (i.e. "Tensor") an index in the list (i.e. ":0") or a type and
    index (i.e. "Tensor:0").

    Similarly, flag_transforms matches its keys in the same way and maps to
    a list of KernelValueConversion constants that make up arg/return specific
    conversion flags.

    Returns:
      - An ods dag list of (ods_name, ods_type) tuples
      - List of (torch_type, [conversion_flag]) for specifying conversions.
    """
    # Make sure any override types are sane.
    if override_types:
      assert len(override_types) == len(siglist), (
          "Mismatch override and actual types")
    # Generate to ods dag list.
    ods_dag_list = []
    for i, sigitem in enumerate(siglist):
      if i in drop_indices:
        # Do not emit in ODS.
        continue
      torch_name = sigitem["name"]
      torch_type = (sigitem["type"]
                    if override_types is None else override_types[i])
      # Look up the type transform.
      ods_type = _first_non_none(type_transforms.get(f"{torch_type}:{i}"),
                                 type_transforms.get(f":{i}"),
                                 type_transforms.get(torch_type))
      if not ods_type:
        raise ValueError(f"Signature item {i}, type {torch_type} did not match "
                         f"a type transform {type_transforms}")
      ods_dag_list.append((torch_name, ods_type))

    # Generate the type conversion flags.
    type_flag_list = []
    for i, sigitem in enumerate(siglist):
      torch_type = (sigitem["type"]
                    if override_types is None else override_types[i])
      # Look up the type transform.
      if i in drop_indices:
        flags = ["kDrop"]
      else:
        flags = _first_non_none(flag_transforms.get(f"{torch_type}:{i}"),
                                flag_transforms.get(f":{i}"),
                                flag_transforms.get(torch_type))
        if flags is None:
          flags = []
      type_flag_list.append((torch_type, flags))
    return ods_dag_list, type_flag_list


class InflightOpDef:

  def __init__(self,
               g: OpGenerator,
               kernel_sig: str,
               ods_name: str,
               op_name: str,
               traits: Sequence[str] = (),
               alias_kernel_names: Sequence[str] = (),
               promote_trailing_out_tensor: bool = False,
               override_arg_types: Sequence[str] = None,
               override_return_types: Sequence[str] = None,
               drop_arg_indices: Sequence[int] = (),
               drop_return_indices: Sequence[int] = ()):
    super().__init__()
    self.g = g
    self.kernel_sig = kernel_sig
    self.ods_name = ods_name
    self.op_name = op_name
    self.traits = list(traits)
    self.alias_kernel_names = list(alias_kernel_names)
    self.promote_trailing_out_tensor = promote_trailing_out_tensor
    self.override_arg_types = override_arg_types
    self.override_return_types = override_return_types
    self.drop_arg_indices = drop_arg_indices
    self.drop_return_indices = drop_return_indices
    self.reg_record = g.get_reg_record(self.kernel_sig)
    self._emitted = False
    self._traceback = traceback.extract_stack()[0:-2]

    # Arg and flag transform dicts.
    self.arg_type_transforms = dict()
    self.arg_flag_transforms = dict()
    self.return_type_transforms = dict()
    self.return_flag_transforms = dict()

    # Signature mapping.
    self._sigs_mapped = False
    self.ods_ins = None
    self.ods_outs = None
    self.arg_type_flags = None
    self.return_type_flags = None

  def __del__(self):
    if not self._emitted:
      print("WARNING: Op defined but not emitted. Defined at:", file=sys.stderr)
      for line in traceback.format_list(self._traceback):
        sys.stderr.write(line)

  def transforms(
      self,
      type_transforms: Dict[str, str] = None,
      flag_transforms: Dict[str, List[str]] = None) -> "InflightOpDef":
    self.arg_transforms(type_transforms=type_transforms,
                        flag_transforms=flag_transforms)
    self.return_transforms(type_transforms=type_transforms,
                           flag_transforms=flag_transforms)
    return self

  def arg_transforms(
      self,
      type_transforms: Dict[str, str] = None,
      flag_transforms: Dict[str, List[str]] = None) -> "InflightOpDef":
    """Adds arg type and flag transforms dicts."""
    if type_transforms:
      self.arg_type_transforms.update(type_transforms)
    if flag_transforms:
      self.arg_flag_transforms.update(flag_transforms)
    return self

  def return_transforms(
      self,
      type_transforms: Dict[str, str] = None,
      flag_transforms: Dict[str, List[str]] = None) -> "InflightOpDef":
    """Adds return type and flag transform dicts."""
    if type_transforms:
      self.return_type_transforms.update(type_transforms)
    if flag_transforms:
      self.return_flag_transforms.update(flag_transforms)
    return self

  def map_signatures(self) -> "InflightOpDef":
    assert not self._sigs_mapped, "Signatures already mapped"
    self._sigs_mapped = True
    self.ods_ins, self.arg_type_flags = self.g._map_sigtypes(
        self.reg_record["arguments"],
        type_transforms=self.arg_type_transforms,
        flag_transforms=self.arg_flag_transforms,
        override_types=self.override_arg_types,
        drop_indices=self.drop_arg_indices)
    self.ods_outs, self.return_type_flags = self.g._map_sigtypes(
        self.reg_record["returns"],
        type_transforms=self.return_type_transforms,
        flag_transforms=self.return_flag_transforms,
        override_types=self.override_return_types,
        drop_indices=self.drop_return_indices)
    return self

  def emit(self):
    assert not self._emitted, "Op already emitted"
    self._emitted = True
    if not self._sigs_mapped:
      self.map_signatures()
    self.g.ods_emitter.emit_opdef(self.ods_name,
                                  self.op_name,
                                  self.reg_record,
                                  ods_ins=self.ods_ins,
                                  ods_outs=self.ods_outs,
                                  traits=self.traits)
    self.g.impl_emitter.emit_kernel_methods(
        self.ods_name,
        self.reg_record,
        arg_type_flags=self.arg_type_flags,
        return_type_flags=self.return_type_flags,
        promote_trailing_out_tensor=self.promote_trailing_out_tensor,
        alias_kernel_names=self.alias_kernel_names)


class EmitterBase:
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

  def indent_sequence(
      self,
      sequence: Sequence[str],
      base_indent_level: int = 0,
      sequence_indent_level: int = 1,
      brackets: Tuple[str, str] = ("[", "]")
  ) -> str:
    base_indent = self._INDENT * base_indent_level
    sequence_indent = f"{base_indent}{self._INDENT * sequence_indent_level}"
    return "".join([
        f"{brackets[0]}\n{sequence_indent}",
        f",\n{sequence_indent}".join(sequence),
        f"\n{base_indent}{brackets[1]}",
    ])

  def print(self, s: str, *, end: str = "\n", indent: bool = True):
    if indent and self.indent_level:
      self.out.write(self._INDENT * self.indent_level)
    self.out.write(s)
    self.out.write(end)

  def quote(self, s: str) -> str:
    s = s.replace(r'"', r'\\"')
    return f'"{s}"'

  def quote_multiline_docstring(self, s: str, indent_level: int = 0) -> str:
    # TODO: Possibly find a python module to markdown the docstring for better
    # document generation.
    # Unlikely to contain the delimitter and since just a docstring, be safe.
    s = s.replace("}]", "")
    # Strip each line.
    s = "\n".join([l.rstrip() for l in s.splitlines()])
    indent = self._INDENT * indent_level
    s = textwrap.indent(s, indent + self._INDENT)
    return "[{\n" + s + "\n" + indent + "}]"


class OdsEmitter(EmitterBase):
  ods_def_prefix = "aten_"
  ods_def_suffix = ""
  ods_template_name = "aten_Op"

  def emit_opdef(self,
                 ods_def_name: str,
                 mnemonic: str,
                 reg_record: REG_OP_TYPE,
                 ods_ins: List[Tuple[str, str]],
                 ods_outs: List[Tuple[str, str]],
                 traits: Sequence[str] = (),
                 summary: Optional[str] = None):
    # Def first-line.
    full_traits = list(traits)
    full_traits.append(
        "DeclareOpInterfaceMethods<TorchBuildableKernelOpInterface>")
    full_traits.append("DeclareOpInterfaceMethods<TorchKernelOpInterface>")
    identifier = f"{self.ods_def_prefix}{ods_def_name}{self.ods_def_suffix}"
    self.print(f"def {identifier}: {self.ods_template_name}"
               f"<{self.quote(mnemonic)}, "
               f"{self.indent_sequence(full_traits, base_indent_level=1)}"
               f"> {{")

    with self.indent():
      # Summary.
      if not summary:
        summary = f"Recognized op for kernel {reg_record['name'][0]}"
      self.print(f"let summary = {self.quote(summary)};")
      # Arguments.
      self.print("let arguments = (ins")
      with self.indent():
        self._emit_dag_list_body(ods_ins)
      self.print(");")

      # Results.
      self.print("let results = (outs")
      with self.indent():
        self._emit_dag_list_body(ods_outs)
      self.print(");")

    # Def last-line.
    self.print("}\n")

  def _emit_dag_list_body(self, items: List[Tuple[str, str]]):
    """Emits a dag of (name, type) pairs."""
    for index, (ods_name, ods_type) in enumerate(items):
      is_last = index == len(items) - 1
      ods_namespec = f":${ods_name}" if ods_name else ""
      self.print(f"{ods_type}{ods_namespec}", end="\n" if is_last else ",\n")


class CCImplEmitter(EmitterBase):

  def emit_kernel_methods(self,
                          ods_def_name: str,
                          reg_record: REG_OP_TYPE,
                          arg_type_flags: List[Tuple[str, List[Tuple[str]]]],
                          return_type_flags: List[Tuple[str, List[Tuple[str]]]],
                          promote_trailing_out_tensor: bool = False,
                          alias_kernel_names: Sequence[str] = ()):
    # getTorchKernelMetadata() method.
    self.print(
        f"Torch::KernelMetadata {ods_def_name}::getTorchKernelMetadata() {{")
    with self.indent():
      self.print("return getTorchBuildKernelMetadata();")
    self.print("}\n")

    # getTorchBuildKernelMetadata() method.
    kernel_name = reg_record["name"][0]
    self.print(
        f"const Torch::BuildKernelMetadata &{ods_def_name}::getTorchBuildKernelMetadata() {{"
    )
    with self.indent():
      self.print("using KVC = Torch::KernelValueConversion::BitMask;")
      self.print("static Torch::BuildKernelMetadata metadata = ([]() {")
      with self.indent():
        self.print("Torch::BuildKernelMetadata m;")
        self.print(f"m.kernelName = {self.quote(kernel_name)};")
        for alias_kernel_name in alias_kernel_names:
          self.print(
              f"m.aliasKernelNames.push_back({self.quote(alias_kernel_name)});")
        if promote_trailing_out_tensor:
          self.print("m.promoteTrailingOutTensor = true;")
        # Arg types/flags.
        arg_types = self._format_cpp_str_initlist(
            [t[0] for t in arg_type_flags])
        self.print(f"m.addArgTypes({arg_types});")
        arg_flags = self._format_cpp_kvc_initlist(
            [t[1] for t in arg_type_flags])
        self.print(f"m.addArgConversions({arg_flags});")
        # Returns types/flags.
        ret_types = self._format_cpp_str_initlist(
            [t[0] for t in return_type_flags])
        self.print(f"m.addReturnTypes({ret_types});")
        ret_flags = self._format_cpp_kvc_initlist(
            [t[1] for t in return_type_flags])
        self.print(f"m.addReturnConversions({ret_flags});")
        self.print("return m;")
      self.print("})();")
      self.print("return metadata;")
    self.print("}\n")

  def _format_cpp_str_initlist(self, strings: Sequence[str]) -> str:
    quoted = [self.quote(s) for s in strings]
    joined = ", ".join(quoted)
    return "{" + joined + "}"

  def _format_cpp_kvc_initlist(self,
                               const_name_lists: List[List[Tuple[str]]]) -> str:

    def or_flags(flag_names: List[Tuple[str]]):
      if not flag_names:
        return "KVC::kNone"
      return "|".join([f"KVC::{n}" for n in flag_names])

    or_d = [or_flags(l) for l in const_name_lists]
    joined = ", ".join(or_d)
    return "{" + joined + "}"


def snakecase_to_camelcase(ident: str) -> str:
  return "".join(x.capitalize() or "_" for x in re.split(r"[\._]", ident))


def _first_non_none(*args) -> Union[None, Any]:
  for arg in args:
    if arg is not None:
      return arg
  return None


def _load_ops_as_dict() -> Dict[str, REG_OP_TYPE]:
  # Returns a list of dicts, each with a name that is a tuple of the form:
  #   (kernel_signature, variant)
  # The kernel signature is a reified form of the argument type signature
  # used throughout PyTorch:
  #   namespace::kernel_name(type1,type2)
  def reify_signature(reg_op):
    kernel_name, unused_variant = reg_op["name"]
    arg_types = [arg["type"] for arg in reg_op["arguments"]]
    return f"{kernel_name}({','.join(arg_types)})"

  reg_ops_list = get_registered_ops()
  return {reify_signature(reg_op): reg_op for reg_op in reg_ops_list}


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
    "//",
    "// Operation summaries and descriptions were systematically derived from public",
    "// API docstrings and are licensed accordingly:",
    "//   https://github.com/pytorch/pytorch/blob/master/LICENSE",
    "//===----------------------------------------------------------------------===//",
    "// This file is automatically generated.  Please do not edit.",
    "// Generated via:",
    f"//   python -m {_get_main_module_name()}",
    "//===----------------------------------------------------------------------===//",
    "",
    "",
])

CC_IMPL_BANNER = "\n".join([
    "//===-------------------------------------------------------------*- cc -*-===//",
    "//",
    "// This file is licensed under the Apache License v2.0 with LLVM Exceptions.",
    "// See https://llvm.org/LICENSE.txt for license information.",
    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception", "//",
    "// Operation summaries and descriptions were systematically derived from public",
    "// API docstrings and are licensed accordingly:",
    "//   https://github.com/pytorch/pytorch/blob/master/LICENSE",
    "//===----------------------------------------------------------------------===//",
    "// This file is automatically generated.  Please do not edit.",
    "// Generated via:", f"//   python -m {_get_main_module_name()}",
    "//===----------------------------------------------------------------------===//",
    "", "", "// clang-format off"
])

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  parser = _create_argparse()
  args = parser.parse_args()
  main(args)
