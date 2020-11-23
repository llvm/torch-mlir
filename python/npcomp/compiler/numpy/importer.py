#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Importers for populating MLIR from AST.
"""
import ast
import sys
import traceback

from _npcomp.mlir import ir

from ..utils import logging
from .interfaces import *

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
      "environment",
  ]

  def __init__(self, ir_c, ir_f, ir_h, filename_ident, environment):
    self.ir_c = ir_c
    self.ir_f = ir_f
    self.ir_h = ir_h
    self.filename_ident = filename_ident
    self.environment = environment

  def abort(self, message):
    """Emits an error diagnostic and raises an exception to abort."""
    loc = self.current_loc
    ir.emit_error(loc, message)
    raise EmittedError(loc, message)

  def check_partial_evaluated(self, result: PartialEvalResult):
    """Checks that a PartialEvalResult has evaluated without error."""
    if result.type == PartialEvalType.ERROR:
      exc_type, exc_value, tb = result.yields
      loc = self.current_loc
      if issubclass(exc_type, UserReportableError):
        message = exc_value.message
      else:
        message = ("Error while evaluating value from environment:\n" +
                   "".join(traceback.format_exception(exc_type, exc_value, tb)))
      ir.emit_error(loc, message)
      raise EmittedError(loc, message)
    if result.type == PartialEvalType.NOT_EVALUATED:
      self.abort("Unable to evaluate expression")

  @property
  def current_loc(self):
    return self.ir_h.builder.current_loc

  def update_loc(self, ast_node):
    self.ir_h.builder.set_file_line_col(self.filename_ident, ast_node.lineno,
                                        ast_node.col_offset)

  def lookup_name(self, name) -> NameReference:
    """Lookup a name in the environment, requiring it to have evaluated."""
    ref = self.environment.resolve_name(name)
    if ref is None:
      self.abort("Could not resolve referenced name '{}'".format(name))
    logging.debug("Map name({}) -> {}", name, ref)
    return ref

  def emit_const_value(self, py_value) -> ir.Value:
    """Codes a value as a constant, returning an ir Value."""
    env = self.environment
    result = env.code_py_value_as_const(py_value)
    if result is NotImplemented:
      self.abort("Cannot code python value as constant: {}".format(py_value))
    return result

  def emit_partial_eval_result(self,
                               partial_result: PartialEvalResult) -> ir.Value:
    """Emits a partial eval result either as a direct IR value or a constant."""
    self.check_partial_evaluated(partial_result)
    if partial_result.type == PartialEvalType.YIELDS_IR_VALUE:
      # Return directly.
      return partial_result.yields
    elif partial_result.type == PartialEvalType.YIELDS_LIVE_VALUE:
      # Import constant.
      return self.emit_const_value(partial_result.yields.live_value)
    else:
      self.abort("Unhandled partial eval result type {}".format(partial_result))


class BaseNodeVisitor(ast.NodeVisitor):
  """Base class of a node visitor that aborts on unhandled nodes."""
  IMPORTER_TYPE = "<unknown>"
  __slots__ = [
      "fctx",
  ]

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
  __slots__ = [
      "ast_fd",
      "_last_was_return",
  ]

  def __init__(self, fctx, ast_fd):
    super().__init__(fctx)
    self.ast_fd = ast_fd
    self._last_was_return = False

  def import_body(self):
    ir_h = self.fctx.ir_h
    for ast_stmt in self.ast_fd.body:
      self._last_was_return = False
      logging.debug("STMT: {}", ast.dump(ast_stmt, include_attributes=True))
      self.visit(ast_stmt)
    if not self._last_was_return:
      # Add a default terminator.
      none_value = ir_h.basicpy_singleton_op(ir_h.basicpy_NoneType).result
      none_cast = ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                               none_value).result
      ir_h.return_op([none_cast])

  def visit_Assign(self, ast_node):
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    for target in ast_node.targets:
      self.fctx.update_loc(target)
      if not isinstance(target.ctx, ast.Store):
        # TODO: Del, AugStore, etc
        self.fctx.abort("Unsupported assignment context type %s" %
                        target.ctx.__class__.__name__)
      name_ref = self.fctx.lookup_name(target.id)
      try:
        name_ref.store(self.fctx.environment, expr.value)
        logging.debug("STORE: {} <- {}", name_ref, expr.value)
      except NotImplementedError:
        self.fctx.abort(
            "Cannot assign to '{}': Store not supported".format(name_ref))

  def visit_Expr(self, ast_node):
    ir_h = self.fctx.ir_h
    _, ip = ir_h.basicpy_exec_op()
    # Evaluate the expression in the exec body.
    orig_ip = ir_h.builder.insertion_point
    ir_h.builder.insertion_point = ip
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    ir_h.basicpy_exec_discard_op([expr.value])
    ir_h.builder.insertion_point = orig_ip

  def visit_Pass(self, ast_node):
    pass

  def visit_Return(self, ast_node):
    ir_h = self.fctx.ir_h
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    casted = ir_h.basicpy_unknown_cast_op(ir_h.basicpy_UnknownType,
                                          expr.value).result
    ir_h.return_op([casted])
    self._last_was_return = True


class ExpressionImporter(BaseNodeVisitor):
  """Imports expression nodes.

  Visitor methods should either raise an exception or set self.value to the
  IR value that the expression lowers to.
  """
  IMPORTER_TYPE = "expression"
  __slots__ = [
      "value",
  ]

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
    env = self.fctx.environment
    ir_const_value = env.code_py_value_as_const(value)
    if ir_const_value is NotImplemented:
      self.fctx.abort("unknown constant type '%r'" % (value,))
    self.value = ir_const_value

  def visit_Attribute(self, ast_node):
    # Import the attribute's value recursively as a partial eval if possible.
    pe_importer = PartialEvalImporter(self.fctx)
    pe_importer.visit(ast_node)
    if pe_importer.partial_eval_result:
      self.fctx.check_partial_evaluated(pe_importer.partial_eval_result)
      self.value = self.fctx.emit_partial_eval_result(
          pe_importer.partial_eval_result)
      return

    self.fctx.abort("unhandled attribute access mode: {}".format(
        ast.dump(ast_node)))

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
      condition_value = ir_h.basicpy_as_predicate_value_op(next_value).result
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

  def visit_Call(self, ast_node):
    # Evaluate positional args.
    evaluated_args = []
    for raw_arg in ast_node.args:
      evaluated_args.append(self.sub_evaluate(raw_arg))

    # Evaluate keyword args.
    keyword_args = []
    for raw_kw_arg in ast_node.keywords:
      keyword_args.append((raw_kw_arg.arg, self.sub_evaluate(raw_kw_arg.value)))

    # Perform partial evaluation of the callee.
    callee_importer = PartialEvalImporter(self.fctx)
    callee_importer.visit(ast_node.func)
    callee_result = callee_importer.partial_eval_result
    if (callee_result and
        callee_result.type == PartialEvalType.YIELDS_LIVE_VALUE):
      # This is a function known to the compiler. Perform a template call.
      call_result = callee_result.yields.resolve_call(self.fctx.environment,
                                                      evaluated_args,
                                                      keyword_args)
      if call_result.type != PartialEvalType.NOT_EVALUATED:
        # Partial evaluation success.
        self.fctx.check_partial_evaluated(call_result)
        self.value = self.fctx.emit_partial_eval_result(call_result)
        return

    # The function is not known to the compiler.
    self.fctx.check_partial_evaluated(callee_result)
    # TODO: Implement first class functions.
    self.fctx.abort("unhandled (potentially first-class function): {}".format(
        ast.dump(ast_node)))

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
    test_result = ir_h.basicpy_as_predicate_value_op(self.sub_evaluate(
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
    name_ref = self.fctx.lookup_name(ast_node.id)
    pe_result = name_ref.load(self.fctx.environment)
    logging.debug("LOAD {} -> {}", name_ref, pe_result)
    self.value = self.fctx.emit_partial_eval_result(pe_result)

  def visit_UnaryOp(self, ast_node):
    ir_h = self.fctx.ir_h
    op = ast_node.op
    operand_value = self.sub_evaluate(ast_node.operand)
    if isinstance(op, ast.Not):
      # Special handling for logical-not.
      condition_value = ir_h.basicpy_as_predicate_value_op(operand_value).result
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


class PartialEvalImporter(BaseNodeVisitor):
  """Importer for performing greedy partial evaluation.

  Concretely this is used for Attribute.value and Call resolution.

  Attribute resolution is not just treated as a normal expression because it
  is first subject to "partial evaluation", allowing the environment's partial
  eval hook to operate on live python values from the containing
  environment versus naively emitting code for attribute resolution for
  entities that can/should be considered constants from the hosting context.
  This is used, for example, to resolve attributes from modules without
  immediately dereferencing/transforming the intervening chain of attributes.
  """
  IMPORTER_TYPE = "partial_eval"
  __slots__ = [
      "partial_eval_result",
  ]

  def __init__(self, fctx):
    super().__init__(fctx)
    self.partial_eval_result = None

  def visit_Attribute(self, ast_node):
    # Sub-evaluate the 'value'.
    sub_eval = PartialEvalImporter(self.fctx)
    sub_eval.visit(ast_node.value)

    if sub_eval.partial_eval_result:
      # Partial sub-evaluation successful.
      sub_result = sub_eval.partial_eval_result
    else:
      # Need to evaluate it as an expression.
      sub_expr = ExpressionImporter(self.fctx)
      sub_expr.visit(ast_node.value)
      assert sub_expr.value, (
          "Evaluated sub expression did not return a value: %r" %
          (ast_node.value))
      sub_result = PartialEvalResult.yields_ir_value(sub_expr.value)

    # Attempt to perform a static getattr as a partial eval if still operating
    # on a live value.
    self.fctx.check_partial_evaluated(sub_result)
    if sub_result.type == PartialEvalType.YIELDS_LIVE_VALUE:
      logging.debug("STATIC getattr '{}' on {}", ast_node.attr, sub_result)
      getattr_result = sub_result.yields.resolve_getattr(
          self.fctx.environment, ast_node.attr)
      if getattr_result.type != PartialEvalType.NOT_EVALUATED:
        self.fctx.check_partial_evaluated(getattr_result)
        self.partial_eval_result = getattr_result
        return
      # If a non-statically evaluable live value, then convert to a constant
      # and dynamic dispatch.
      ir_value = self.fctx.emit_const_value(sub_result.yields.live_value)
    else:
      ir_value = sub_result.yields

    # Yielding an IR value from a recursive partial evaluation means that the
    # entire chain needs to be hoisted to IR.
    # TODO: Implement.
    self.fctx.abort("dynamic-emitted getattr not yet supported: %r" %
                    (ir_value,))

  def visit_Name(self, ast_node):
    name_ref = self.fctx.lookup_name(ast_node.id)
    partial_eval_result = name_ref.load(self.fctx.environment)
    logging.debug("PARTIAL EVAL {} -> {}", name_ref, partial_eval_result)
    self.partial_eval_result = partial_eval_result
