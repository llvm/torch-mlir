#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates ODS for a registry of ops."""

from typing import TextIO

import argparse
from contextlib import contextmanager
import importlib
import logging
import re
import sys
import textwrap

from .registry import *

_INDENT = "  "


class OdsEmitter:
  ods_prefix = "ATen_"
  ods_suffix = "Op"
  ods_value_template = "ATen_ImmutableTensorOp"
  ods_ref_template = "ATen_RefTensorOp"
  op_prefix = ""

  def __init__(self, r: OpRegistry, out: TextIO):
    super().__init__()
    self.r = r
    self.out = out
    self.indent_level = 0

  def emit_ods(self):
    for op_m in self.r.mappings:
      if isinstance(op_m, SimpleOpMapping):
        self._emit_simple_op_mapping(op_m)
      else:
        logging.warn(f"Unrecognized op mapping type: {op_m!r}")

  def _emit_simple_op_mapping(self, op_m: SimpleOpMapping):
    identifier = (f"{self.ods_prefix}"
                  f"{_snakecase_to_camelcase(op_m.mlir_operation_name)}"
                  f"{self.ods_suffix}")
    traits = []

    if op_m.is_outref_form:
      template_name = self.ods_ref_template
      summary = "See non-inplace op variant."
      description = ""
    else:
      template_name = self.ods_value_template
      summary, description = _split_docstring(op_m.op_f.__doc__)

    if not op_m.is_outref_form:
      traits.append("NoSideEffect")
    self.print(f"def {identifier}: {template_name}"
               f"<{_quote(op_m.mlir_operation_name)}, ["
               f"{', '.join(traits)}"
               f"]> {{")

    # Summary.
    with self.indent():
      self.print(f"let summary = {_quote(summary)};")

    # Arguments.
    with self.indent():
      self.print("let arguments = (ins")
      with self.indent():
        operand_len = len(op_m.operand_map)
        for index, (_, value_spec) in enumerate(op_m.operand_map):
          is_last = index == operand_len - 1
          self.print(f"{value_spec.mlir_ods_predicate}:${value_spec.name}",
                     end="\n" if is_last else ",\n")
      self.print(");")

    # Results (omitted if an outref/inplace form).
    with self.indent():
      if op_m.is_outref_form:
        self.print("let results = (outs);")
      else:
        self.print("let results = (outs")
        with self.indent():
          result_len = len(op_m.result_map)
          for index, (_, value_spec) in enumerate(op_m.result_map):
            is_last = index == result_len - 1
            self.print(f"{value_spec.mlir_ods_predicate}:${value_spec.name}",
                       end="\n" if is_last else ",\n")
        self.print(");")

    # Description and extra class declarations.
    with self.indent():
      if description:
        quoted_description = _quote_multiline_docstring(
            description, indent_level=self.indent_level)
        self.print(f"let description = {quoted_description};")

    self.print("}\n")

  @contextmanager
  def indent(self, level=1):
    self.indent_level += level
    yield
    self.indent_level -= level
    assert self.indent_level >= 0, "Unbalanced indentation"

  def print(self, s, *, end="\n", indent=True):
    if indent and self.indent_level:
      self.out.write(_INDENT * self.indent_level)
    self.out.write(s)
    self.out.write(end)


def _snakecase_to_camelcase(ident: str):
  return "".join(x.capitalize() or "_" for x in re.split(r"[\._]", ident))


def _quote(s: str):
  s = s.replace(r'"', r'\\"')
  return f'"{s}"'


def _quote_multiline_docstring(s: str, indent_level: int = 0):
  # TODO: Possibly find a python module to markdown the docstring for better
  # document generation.
  # Unlikely to contain the delimitter and since just a docstring, be safe.
  s = s.replace("}]", "")
  # Strip each line.
  s = "\n".join([l.rstrip() for l in s.splitlines()])
  indent = _INDENT * indent_level
  s = textwrap.indent(s, indent + _INDENT)
  return "[{\n" + s + "\n" + indent + "}]"


def _split_docstring(docstring: str):
  """Splits the docstring into a summary and description."""
  lines = docstring.splitlines()
  # Skip leading blank lines.
  while lines and not lines[0]:
    lines = lines[1:]
  if len(lines) > 2:
    return lines[0], "\n".join(lines[2:])
  else:
    return lines[0]


def main(args):
  r = OpRegistry()
  # Populate from modules that provide a populate() function.
  op_modules = [args.op_module]
  for m_name in op_modules:
    logging.info(f"Populating from module: {m_name}")
    m = importlib.import_module(m_name, package=__package__)
    f = getattr(m, "populate")
    f(r)

  out = sys.stdout

  # Write file header.
  module_name = sys.modules["__main__"].__loader__.name
  banner_lines = [
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
      f"//   python -m {module_name} {' '.join(sys.argv[1:])}",
      "//===----------------------------------------------------------------------===//",
      "",
      "",
  ]
  banner_lines = [l.strip() for l in banner_lines]
  out.write("\n".join(banner_lines))

  emitter = OdsEmitter(r, out=out)
  emitter.emit_ods()


def _create_argparse():
  parser = argparse.ArgumentParser(prog="generate_ods")
  parser.add_argument(
      "--op_module",
      default=".aten_ops",
      help="Name of a python module for populating the registry")
  return parser


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  parser = _create_argparse()
  args = parser.parse_args()
  main(args)
