#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Meta-representation of a compiled module under construction.

As an MlirModule is built-up for a compiled Python program, instances here
are used to represent it in a user-servicable way. In addition to
inspectability, this also provides update interfaces for type refinement and
other high level tasks to be performed.
"""

import io

from typing import Any, Dict, Protocol, Tuple

from . import types
from mlir import ir as _ir

__all__ = [
    "MetaModule",
    "Symbol",
    "SymbolAtom",
]

SymbolAtom = str
Symbol = Tuple[SymbolAtom]


class ExportNode:
  """An exported entity (function, variable, whatever)."""
  pass


class ExportGenericFunction(ExportNode):
  """An exported 'generic' function.

  Such functions are typically the first step of import where some type
  information may be known but we may still want to hint to the compiler to
  materialize specific specializations and pattern match them at runtime.
  """
  __slots__ = [
      "ir_symbol_name",
      "signature",
  ]

  def __init__(self, ir_symbol_name: str, signature: types.Signature):
    self.ir_symbol_name = ir_symbol_name
    self.signature = signature

  def __repr__(self):
    return f"generic func @{self.ir_symbol_name} signature {self.signature}"


class ExportSpecializedFunction(ExportNode):
  """An exported 'specialized' function.

  Such functions have been fully specialized and cannot be further customized.
  """
  __slots__ = [
      "ir_symbol_name",
      "signature",
  ]

  def __init__(self, ir_symbol_name: str, signature: types.Signature):
    self.ir_symbol_name = ir_symbol_name
    self.signature = signature

  def __repr__(self):
    return f"specialized func @{self.ir_symbol_name} signature {self.signature}"


SymbolTable = Dict[Symbol, ExportNode]


class MetaModule:
  """Wraps an MLIR Module and provides high-level access.

  The meta-module maps a list of symbols to delegate objects that can manipulate
  them at the IR level. Each symbol is a tuple of string names that form a
  tree of exported entities that matches the object-structure of the program
  being extracted.

  This class provides API-level access to these exports. For easier navigation,
  the tree structure is also exposed as dynamic-attribute objects under the
  'exports' property.
  """
  _module: _ir.Module
  _symbol_table: SymbolTable

  __slots__ = [
      "_module",
      "_symbol_table",
  ]

  def __init__(self, module: _ir.Module):
    self._module = module
    self._symbol_table = dict()

  @property
  def module(self) -> _ir.Module:
    return self._module

  @property
  def symbol_table(self) -> SymbolTable:
    return self._symbol_table

  def export_symbol(self, symbol: Symbol, export_node: ExportNode):
    """Exports a symbol into the namespace."""
    assert isinstance(export_node, ExportNode)
    if symbol in self._symbol_table:
      raise ValueError(f"Attempt to export duplicate symbol {symbol}")
    self._symbol_table[symbol] = export_node

  def __repr__(self):
    super_repr = super().__repr__()
    f = io.StringIO()

    def p(*args):
      print(*args, file=f)

    p(f"<MetaModule ({super_repr})>:")
    p("  >> Symbol Table:")
    for symbol, node in sorted(self._symbol_table.items()):
      pretty_symbol = ".".join(symbol)
      node_repr = repr(node).splitlines() or [""]
      p(f"     '{pretty_symbol}' -> {node_repr[0]}:")
      for node_line in node_repr[1:]:
        p(f"       {node_line}")
    p()
    p("  >> MLIR Assembly:")
    mlir_asm = self._module.operation.get_asm(
        large_elements_limit=10).splitlines()
    for asm_line in mlir_asm:
      p(f"     {asm_line}")
    p(f"</end MetaModule>")
    return f.getvalue()
