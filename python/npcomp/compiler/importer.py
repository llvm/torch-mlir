#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Importers for populating MLIR from AST.
"""
import ast
import sys

from _npcomp.mlir import ir

from . import logging

__all__ = [
    "FunctionContext",
    "FunctionDefImporter",
    "ExpressionImporter",
]


class FunctionContext:
  """Accounting information for importing a function."""
  __slots__ = [
      "ir_c",
      "ir_f",
      "ir_h",
      "filename_ident",
      "local_name_value_map",
  ]

  def __init__(self, ir_c, ir_f, ir_h, filename_ident):
    self.ir_c = ir_c
    self.ir_f = ir_f
    self.ir_h = ir_h
    self.filename_ident = filename_ident
    self.local_name_value_map = dict()

  def abort(self, message):
    """Emits an error diagnostic and raises an exception to abort."""
    loc = self.current_loc
    ir.emit_error(loc, message)
    raise EmittedError(loc, message)

  @property
  def current_loc(self):
    return self.ir_h.builder.current_loc

  def update_loc(self, ast_node):
    self.ir_h.builder.set_file_line_col(self.filename_ident, ast_node.lineno,
                                        ast_node.col_offset)

  def map_local_name(self, name, value):
    self.local_name_value_map[name] = value
    logging.debug("Map name({}) -> value({})", name, value)


class BaseNodeVisitor(ast.NodeVisitor):
  """Base class of a node visitor that aborts on unhandled nodes."""
  IMPORTER_TYPE = "<unknown>"

  def __init__(self, fctx):
    super().__init__()
    self.fctx = fctx

  def visit(self, node):
    self.fctx.update_loc(node)
    return super().visit(node)

  def generic_visit(self, ast_node):
    logging.debug("UNHANDLED NODE: {}", ast.dump(ast_node))
    self.fctx.abort("unhandled python %s AST node '%s'" %
                    (self.IMPORTER_TYPE, ast_node.__class__.__name__))


class FunctionDefImporter(BaseNodeVisitor):
  """AST visitor for importing a function's statements.

  Handles nodes that are direct children of a FunctionDef.
  """
  IMPORTER_TYPE = "statement"

  def __init__(self, fctx, ast_fd):
    super().__init__(fctx)
    self.ast_fd = ast_fd

  def import_body(self):
    for ast_stmt in self.ast_fd.body:
      logging.debug("STMT: {}", ast.dump(ast_stmt, include_attributes=True))
      self.visit(ast_stmt)

  def visit_Assign(self, ast_node):
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    for target in ast_node.targets:
      self.fctx.update_loc(target)
      if not isinstance(target.ctx, ast.Store):
        # TODO: Del, AugStore, etc
        self.fctx.abort("Unsupported assignment context type %s" %
                        target.ctx.__class__.__name__)
      self.fctx.map_local_name(target.id, expr.value)

  def visit_Return(self, ast_node):
    ir_h = self.fctx.ir_h
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    casted = ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                          expr.value).result
    ir_h.return_op([casted])


class ExpressionImporter(BaseNodeVisitor):
  IMPORTER_TYPE = "expression"

  def __init__(self, fctx):
    super().__init__(fctx)
    self.value = None

  def visit(self, node):
    super().visit(node)
    assert self.value, ("ExpressionImporter did not assign a value (%r)" %
                        (ast.dump(node),))

  def sub_evaluate(self, sub_node):
    sub_importer = ExpressionImporter(self.fctx)
    sub_importer.visit(sub_node)
    return sub_importer.value

  def emit_constant(self, value):
    ir_c = self.fctx.ir_c
    ir_h = self.fctx.ir_h
    if value is True:
      self.value = ir_h.basicpy_bool_constant_op(True).result
    elif value is False:
      self.value = ir_h.basicpy_bool_constant_op(False).result
    elif value is None:
      self.value = ir_h.basicpy_singleton_op(ir_h.basicpy_NoneType).result
    elif isinstance(value, int):
      # TODO: Configurable type mapping
      ir_type = ir_h.i64_type
      ir_attr = ir_c.integer_attr(ir_type, value)
      self.value = ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(value, float):
      # TODO: Configurable type mapping
      ir_type = ir_h.f64_type
      ir_attr = ir_c.float_attr(ir_type, value)
      self.value = ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(value, str):
      self.value = ir_h.basicpy_str_constant_op(value).result
    elif isinstance(value, bytes):
      self.value = ir_h.basicpy_bytes_constant_op(value).result
    elif isinstance(value, type(...)):
      self.value = ir_h.basicpy_singleton_op(ir_h.basicpy_EllipsisType).result
    else:
      self.fctx.abort("unknown constant type '%r'" % (value,))

  def visit_BinOp(self, ast_node):
    ir_h = self.fctx.ir_h
    left = self.sub_evaluate(ast_node.left)
    right = self.sub_evaluate(ast_node.right)
    self.value = ir_h.basicpy_binary_expr_op(
        ir_h.basicpy_UnknownType, left, right,
        ast_node.op.__class__.__name__).result

  def visit_BoolOp(self, ast_node):
    ir_h = self.fctx.ir_h
    if isinstance(ast_node.op, ast.And):
      return_first_true = False
    elif isinstance(ast_node.op, ast.Or):
      return_first_true = True
    else:
      self.fctx.abort("unknown bool op %r" % (ast.dump(ast_node.op)))

    def emit_next(next_nodes):
      next_node = next_nodes[0]
      next_nodes = next_nodes[1:]
      next_value = self.sub_evaluate(next_node)
      if not next_nodes:
        return next_value
      condition_value = ir_h.basicpy_to_boolean_op(next_value).result
      if_op, then_ip, else_ip = ir_h.scf_if_op([ir_h.basicpy_UnknownType],
                                               condition_value, True)
      orig_ip = ir_h.builder.insertion_point
      # Short-circuit return case.
      ir_h.builder.insertion_point = then_ip if return_first_true else else_ip
      next_value_casted = ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                                       next_value).result
      ir_h.scf_yield_op([next_value_casted])
      # Nested evaluate next case.
      ir_h.builder.insertion_point = else_ip if return_first_true else then_ip
      nested_value = emit_next(next_nodes)
      nested_value_casted = next_value_casted = ir_h.basicpy_unknown_cast_op(
          ir_h.basicpy_UnknownType, nested_value).result
      ir_h.scf_yield_op([nested_value_casted])
      ir_h.builder.insertion_point = orig_ip
      return if_op.result

    self.value = emit_next(ast_node.values)

  def visit_Compare(self, ast_node):
    # Short-circuit comparison (degenerates to binary comparison when just
    # two operands).
    ir_h = self.fctx.ir_h
    false_value = ir_h.basicpy_bool_constant_op(False).result

    def emit_next(left_value, comparisons):
      operation, right_node = comparisons[0]
      comparisons = comparisons[1:]
      right_value = self.sub_evaluate(right_node)
      compare_result = ir_h.basicpy_binary_compare_op(
          left_value, right_value, operation.__class__.__name__).result
      # Terminate by yielding the final compare result.
      if not comparisons:
        return compare_result

      # Emit 'if' op and recurse. The if op takes an i1 (core dialect
      # requirement) and returns a basicpy.BoolType. Since this is an 'and',
      # all else clauses yield a false value.
      compare_result_i1 = ir_h.basicpy_bool_cast_op(ir_h.i1_type,
                                                    compare_result).result
      if_op, then_ip, else_ip = ir_h.scf_if_op([ir_h.basicpy_BoolType],
                                               compare_result_i1, True)
      orig_ip = ir_h.builder.insertion_point
      # Build the else clause.
      ir_h.builder.insertion_point = else_ip
      ir_h.scf_yield_op([false_value])
      # Build the then clause.
      ir_h.builder.insertion_point = then_ip
      nested_result = emit_next(right_value, comparisons)
      ir_h.scf_yield_op([nested_result])
      ir_h.builder.insertion_point = orig_ip
      return if_op.result

    self.value = emit_next(self.sub_evaluate(ast_node.left),
                           list(zip(ast_node.ops, ast_node.comparators)))

  def visit_IfExp(self, ast_node):
    ir_h = self.fctx.ir_h
    test_result = ir_h.basicpy_to_boolean_op(self.sub_evaluate(
        ast_node.test)).result
    if_op, then_ip, else_ip = ir_h.scf_if_op([ir_h.basicpy_UnknownType],
                                             test_result, True)

    orig_ip = ir_h.builder.insertion_point
    # Build the then clause
    ir_h.builder.insertion_point = then_ip
    then_result = self.sub_evaluate(ast_node.body)
    ir_h.scf_yield_op([
        ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                     then_result).result
    ])
    # Build the then clause.
    ir_h.builder.insertion_point = else_ip
    orelse_result = self.sub_evaluate(ast_node.orelse)
    ir_h.scf_yield_op([
        ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                     orelse_result).result
    ])
    ir_h.builder.insertion_point = orig_ip

    self.value = if_op.result

  def visit_Name(self, ast_node):
    if not isinstance(ast_node.ctx, ast.Load):
      self.fctx.abort("Unsupported expression name context type %s" %
                      ast_node.ctx.__class__.__name__)
    # TODO: Need to apply scope rules: local, global, ...
    value = self.fctx.local_name_value_map.get(ast_node.id)
    if value is None:
      self.fctx.abort("Local variable '%s' has not been assigned" % ast_node.id)
    self.value = value

  def visit_UnaryOp(self, ast_node):
    ir_h = self.fctx.ir_h
    op = ast_node.op
    operand_value = self.sub_evaluate(ast_node.operand)
    if isinstance(op, ast.Not):
      # Special handling for logical-not.
      condition_value = ir_h.basicpy_to_boolean_op(operand_value).result
      true_value = ir_h.basicpy_bool_constant_op(True).result
      false_value = ir_h.basicpy_bool_constant_op(False).result
      self.value = ir_h.select_op(condition_value, false_value,
                                  true_value).result
    else:
      self.fctx.abort("Unknown unary op %r", (ast.dump(op)))

  if sys.version_info < (3, 8, 0):
    # <3.8 breaks these out into separate AST classes.
    def visit_Num(self, ast_node):
      self.emit_constant(ast_node.n)

    def visit_Str(self, ast_node):
      self.emit_constant(ast_node.s)

    def visit_Bytes(self, ast_node):
      self.emit_constant(ast_node.s)

    def visit_NameConstant(self, ast_node):
      self.emit_constant(ast_node.value)

    def visit_Ellipsis(self, ast_node):
      self.emit_constant(...)
  else:

    def visit_Constant(self, ast_node):
      self.emit_constant(ast_node.value)


class EmittedError(Exception):
  """Exception subclass that indicates an error diagnostic has been emitted.

  By throwing, this lets us abort and handle at a higher level so as not
  to duplicate diagnostics.
  """

  def __init__(self, loc, message):
    super().__init__("%s (at %r)" % (message, loc))
    self.loc = loc
