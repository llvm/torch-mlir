#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""General utilities for working with MLIR."""

from typing import Optional, Tuple

from mlir import ir as _ir
from npcomp import _cext

__all__ = [
    "ImportContext",
]


class ImportContext:
  """Simple container for things that we update while importing.

  This is also where we stash various helpers to work around awkward/missing
  MLIR Python API features.
  """
  __slots__ = [
      "context",
      "loc",
      "module",
      "_ip_stack",

      # Cached types.
      "unknown_type",
      "bool_type",
      "bytes_type",
      "ellipsis_type",
      "i1_type",
      "index_type",
      "none_type",
      "str_type",
      "unknown_array_type",
      "unknown_tensor_type",

      # Cached attributes.
      "i1_true",
      "i1_false",
  ]

  def __init__(self, context: Optional[_ir.Context]):
    self.context = _ir.Context() if not context else context
    _cext.register_all_dialects(self.context)

    self.loc = _ir.Location.unknown(context=self.context)  # type: _ir.Location
    self.module = None  # type: Optional[_ir.Module]
    self._ip_stack = []

    # Cache some types and attributes.
    with self.context:
      # Types.
      # TODO: Consolidate numpy.any_dtype and basicpy.UnknownType.
      self.unknown_type = _ir.Type.parse("!basicpy.UnknownType")
      self.bool_type = _ir.Type.parse("!basicpy.BoolType")
      self.bytes_type = _ir.Type.parse("!basicpy.BytesType")
      self.ellipsis_type = _ir.Type.parse("!basicpy.EllipsisType")
      self.none_type = _ir.Type.parse("!basicpy.NoneType")
      self.str_type = _ir.Type.parse("!basicpy.StrType")
      self.i1_type = _ir.IntegerType.get_signless(1)
      self.index_type = _ir.IndexType.get()
      self.unknown_tensor_type = _ir.UnrankedTensorType.get(self.unknown_type,
                                                            loc=self.loc)
      self.unknown_array_type = _cext.shaped_to_ndarray_type(
          self.unknown_tensor_type)

      # Attributes.
      self.i1_true = _ir.IntegerAttr.get(self.i1_type, 1)
      self.i1_false = _ir.IntegerAttr.get(self.i1_type, 0)

  def set_file_line_col(self, file: str, line: int, col: int):
    self.loc = _ir.Location.file(file, line, col, context=self.context)

  def push_ip(self, new_ip: _ir.InsertionPoint):
    self._ip_stack.append(new_ip)

  def pop_ip(self):
    assert self._ip_stack, "Mismatched push_ip/pop_ip: stack is empty on pop"
    del self._ip_stack[-1]

  @property
  def ip(self):
    assert self._ip_stack, "InsertionPoint requested but stack is empty"
    return self._ip_stack[-1]

  def insert_before_terminator(self, block: _ir.Block):
    self.push_ip(_ir.InsertionPoint.at_block_terminator(block))

  def insert_end_of_block(self, block: _ir.Block):
    self.push_ip(_ir.InsertionPoint(block))

  def FuncOp(self, name: str, func_type: _ir.Type,
             create_entry_block: bool) -> Tuple[_ir.Operation, _ir.Block]:
    """Creates a |func| op.

    TODO: This should really be in the MLIR API.
    Returns:
      (operation, entry_block)
    """
    with self.context, self.loc:
      attrs = {
          "type": _ir.TypeAttr.get(func_type),
          "sym_name": _ir.StringAttr.get(name),
      }
      op = _ir.Operation.create("func", regions=1, attributes=attrs, ip=self.ip)
      body_region = op.regions[0]
      entry_block = body_region.blocks.append(*func_type.inputs)
      return op, entry_block

  def basicpy_ExecOp(self):
    """Creates a basicpy.exec op.

    Returns:
      Insertion point to the body.
    """
    op = _ir.Operation.create("basicpy.exec",
                              regions=1,
                              ip=self.ip,
                              loc=self.loc)
    b = op.regions[0].blocks.append()
    return _ir.InsertionPoint(b)

  def basicpy_FuncTemplateCallOp(self, result_type, callee_symbol, args,
                                 arg_names):
    with self.loc, self.ip:
      attributes = {
          "callee":
              _ir.FlatSymbolRefAttr.get(callee_symbol),
          "arg_names":
              _ir.ArrayAttr.get([_ir.StringAttr.get(n) for n in arg_names]),
      }
      op = _ir.Operation.create("basicpy.func_template_call",
                                results=[result_type],
                                operands=args,
                                attributes=attributes,
                                ip=self.ip)
      return op

  def scf_IfOp(self, results, condition: _ir.Value, with_else_region: bool):
    """Creates an SCF if op.

    Returns:
      (if_op, then_ip, else_ip) if with_else_region, otherwise (if_op, then_ip)
    """
    op = _ir.Operation.create("scf.if",
                              results=results,
                              operands=[condition],
                              regions=2 if with_else_region else 1,
                              loc=self.loc,
                              ip=self.ip)
    then_region = op.regions[0]
    then_block = then_region.blocks.append()
    if with_else_region:
      else_region = op.regions[1]
      else_block = else_region.blocks.append()
      return op, _ir.InsertionPoint(then_block), _ir.InsertionPoint(else_block)
    else:
      return op, _ir.InsertionPoint(then_block)

  def scf_YieldOp(self, operands):
    return _ir.Operation.create("scf.yield",
                                operands=operands,
                                loc=self.loc,
                                ip=self.ip)
