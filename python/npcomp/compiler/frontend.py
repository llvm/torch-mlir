#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Frontend to the compiler, allowing various ways to import code.
"""

import ast
import inspect
import sys

from _npcomp.mlir import ir
from npcomp.dialect import Numpy

from . import logging

__all__ = [
    "ImportFrontend",
]


class ImportFrontend:
  """Frontend for importing various entities into a Module."""

  def __init__(self, ir_context: ir.MLIRContext = None):
    self._ir_context = ir.MLIRContext() if not ir_context else ir_context
    self._ir_module = self._ir_context.new_module()
    self._helper = Numpy.DialectHelper(self._ir_context)

  @property
  def ir_context(self):
    return self._ir_context

  @property
  def ir_module(self):
    return self._ir_module

  @property
  def ir_h(self):
    return self._helper

  def import_global_function(self, f):
    """Imports a global function.

    This facility is not general and does not allow customization of the
    containing environment, method import, etc.

    Most errors are emitted via the MLIR context's diagnostic infrastructure,
    but errors related to extracting source, etc are raised directly.

    Args:
      f: The python callable.
    """
    h = self.ir_h
    ir_c = self.ir_context
    ir_m = self.ir_module
    filename = inspect.getsourcefile(f)
    source_lines, start_lineno = inspect.getsourcelines(f)
    source = "".join(source_lines)
    ast_root = ast.parse(source, filename=filename)
    ast.increment_lineno(ast_root, start_lineno - 1)
    ast_fd = ast_root.body[0]
    filename_ident = ir_c.identifier(filename)

    # Define the function.
    # TODO: Much more needs to be done here (arg/result mapping, etc)
    logging.debug(":::::::")
    logging.debug("::: Importing global function {}:\n{}", ast_fd.name,
                  ast.dump(ast_fd, include_attributes=True))
    h.builder.set_file_line_col(filename_ident, ast_fd.lineno,
                                ast_fd.col_offset)
    h.builder.insert_before_terminator(ir_m.first_block)
    ir_f_type = h.function_type([], [h.basicpy_UnknownType])
    ir_f = h.func_op(ast_fd.name, ir_f_type, create_entry_block=True)

    fctx = FunctionContext(ir_c=ir_c,
                           ir_f=ir_f,
                           ir_h=h,
                           filename_ident=filename_ident)
    fdimport = FunctionDefImporter(fctx, ast_fd)
    fdimport.import_body()
    return ir_f


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
    casted = ir_h.basicpy_unknown_cast(ir_h.basicpy_UnknownType,
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

  def visit_Constant(self, ast_node):
    ir_c = self.fctx.ir_c
    ir_h = self.fctx.ir_h
    if isinstance(ast_node, ast.Num):
      # Handle numeric constants.
      nval = ast_node.n
      if isinstance(nval, int):
        # TODO: Configurable type mapping
        ir_type = ir_h.i64_type
        ir_attr = ir_c.integer_attr(ir_type, nval)
      elif isinstance(nval, float):
        # TODO: Configurable type mapping
        ir_type = ir_h.f64_type
        ir_attr = ir_c.float_attr(ir_type, nval)
      else:
        self.fctx.abort("unsupported numeric constant type: %r" % (nval,))
      self.value = ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(ast_node, ast.NameConstant):
      if ast_node.value is True:
        self.value = ir_h.basicpy_bool_constant_op(True).result
      elif ast_node.value is False:
        self.value = ir_h.basicpy_bool_constant_op(False).result
      else:
        self.fctx.abort("unknown named constant '%r'" % (ast_node.value,))
    else:
      self.fctx.abort("unknown constant type %s" %
                      (ast_node.__class__.__name__))

  def visit_BinOp(self, ast_node):
    ir_c = self.fctx.ir_c
    ir_h = self.fctx.ir_h
    left = ExpressionImporter(self.fctx)
    left.visit(ast_node.left)
    right = ExpressionImporter(self.fctx)
    right.visit(ast_node.right)
    ir_attrs = ir_c.dictionary_attr(
        {"operation": ir_c.string_attr(ast_node.op.__class__.__name__)})
    self.fctx.update_loc(ast_node)
    # TODO: Change to a registered op.
    self.value = ir_h.op("basicpy.binary_expr", [ir_h.basicpy_UnknownType],
                         [left.value, right.value], ir_attrs).result

  def visit_Name(self, ast_node):
    if not isinstance(ast_node.ctx, ast.Load):
      self.fctx.abort("Unsupported expression name context type %s" %
                      ast_node.ctx.__class__.__name__)
    # TODO: Need to apply scope rules: local, global, ...
    value = self.fctx.local_name_value_map.get(ast_node.id)
    if value is None:
      self.fctx.abort("Local variable '%s' has not been assigned" % ast_node.id)
    self.value = value

  if sys.version_info < (3, 8, 0):
    visit_Num = visit_Constant
    visit_Str = visit_Constant
    visit_Bytes = visit_Constant
    visit_NameConstant = visit_Constant
    visit_Ellipsis = visit_Constant
  else:
    # For >= 3.8.0, these are deprecated but still may be called for
    # compatibility (in addition to visit_Constant). Just make them no-op.
    def ignore(self, ast_node):
      pass

    visit_Num = ignore
    visit_Str = ignore
    visit_Bytes = ignore
    visit_NameConstant = ignore
    visit_Ellipsis = ignore


class EmittedError(Exception):
  """Exception subclass that indicates an error diagnostic has been emitted.

  By throwing, this lets us abort and handle at a higher level so as not
  to duplicate diagnostics.
  """

  def __init__(self, loc, message):
    super().__init__("%s (at %r)" % (message, loc))
    self.loc = loc
