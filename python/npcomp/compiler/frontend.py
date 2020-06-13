#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Frontend to the compiler, allowing various ways to import code.
"""

import ast
import inspect

from _npcomp.mlir import ir
from _npcomp.mlir.dialect import ScfDialectHelper
from npcomp.dialect import Numpy

from . import logging
from .importer import *
from .target import *

__all__ = [
    "ImportFrontend",
]


# TODO: Remove this hack in favor of a helper function that combines
# multiple dialect helpers so that we don't need to deal with the sharp
# edge of initializing multiple native base classes.
class AllDialectHelper(Numpy.DialectHelper, ScfDialectHelper):

  def __init__(self, *args, **kwargs):
    Numpy.DialectHelper.__init__(self, *args, **kwargs)
    ScfDialectHelper.__init__(self, *args, **kwargs)


class ImportFrontend:
  """Frontend for importing various entities into a Module."""
  __slots__ = [
      "_ir_context",
      "_ir_module",
      "_helper",
      "_target_factory",
  ]

  def __init__(self,
               ir_context: ir.MLIRContext = None,
               target_factory: TargetFactory = GenericTarget64):
    self._ir_context = ir.MLIRContext() if not ir_context else ir_context
    self._ir_module = self._ir_context.new_module()
    self._helper = AllDialectHelper(self._ir_context,
                                    ir.OpBuilder(self._ir_context))
    self._target_factory = target_factory

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
    target = self._target_factory(h)
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

    # TODO: VERY BAD: Assumes all positional params.
    unknown_type = h.basicpy_UnknownType
    f_params = inspect.signature(f).parameters
    arg_count = len(f_params)
    ir_f_type = h.function_type([unknown_type for _ in range(arg_count)],
                                [unknown_type])

    h.builder.set_file_line_col(filename_ident, ast_fd.lineno,
                                ast_fd.col_offset)
    h.builder.insert_before_terminator(ir_m.first_block)
    ir_f = h.func_op(ast_fd.name, ir_f_type, create_entry_block=True)
    fctx = FunctionContext(ir_c=ir_c,
                           ir_f=ir_f,
                           ir_h=h,
                           filename_ident=filename_ident,
                           target=target)
    for f_arg, ir_arg in zip(f_params, ir_f.first_block.args):
      fctx.map_local_name(f_arg, ir_arg)

    fdimport = FunctionDefImporter(fctx, ast_fd)
    fdimport.import_body()
    return ir_f
