#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Frontend to the compiler, allowing various ways to import code.
"""

import ast
import inspect
import textwrap
from typing import Optional

from _npcomp.mlir import ir
from _npcomp.mlir.dialect import ScfDialectHelper
from npcomp.dialect import Numpy

from ..utils import logging
from .importer import *
from .interfaces import *
from .name_resolver_base import *
from .value_coder_base import *
from .target import *

__all__ = [
    "ImportFrontend",
]


class ImportFrontend:
  """Frontend for importing various entities into a Module."""
  __slots__ = [
      "_ir_context",
      "_ir_module",
      "_ir_h",
      "_config",
  ]

  def __init__(self,
               *,
               config: Configuration,
               ir_context: ir.MLIRContext = None):
    super().__init__()
    self._ir_context = ir.MLIRContext() if not ir_context else ir_context
    self._ir_module = self._ir_context.new_module()
    self._ir_h = AllDialectHelper(self._ir_context,
                                  ir.OpBuilder(self._ir_context))
    self._config = config

  @property
  def ir_context(self):
    return self._ir_context

  @property
  def ir_module(self):
    return self._ir_module

  @property
  def ir_h(self):
    return self._ir_h

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
    target = self._config.target_factory(h)
    filename = inspect.getsourcefile(f)
    source_lines, start_lineno = inspect.getsourcelines(f)
    source = "".join(source_lines)
    source = textwrap.dedent(source)
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
    f_signature = inspect.signature(f)
    f_params = f_signature.parameters
    f_input_types = [
        self._resolve_signature_annotation(target, p.annotation)
        for p in f_params.values()
    ]
    f_return_type = self._resolve_signature_annotation(
        target, f_signature.return_annotation)
    ir_f_type = h.function_type(f_input_types, [f_return_type])
    import sys
    print("--->", ir_f_type, file=sys.stderr)
    h.builder.set_file_line_col(filename_ident, ast_fd.lineno,
                                ast_fd.col_offset)
    h.builder.insert_before_terminator(ir_m.first_block)
    # TODO: Do not hardcode this IREE attribute.
    attrs = ir_c.dictionary_attr({"iree.module.export": ir_c.unit_attr})
    ir_f = h.func_op(ast_fd.name,
                     ir_f_type,
                     create_entry_block=True,
                     attrs=attrs)
    env = self._create_const_global_env(f,
                                        parameter_bindings=zip(
                                            f_params.keys(),
                                            ir_f.first_block.args),
                                        target=target)
    fctx = FunctionContext(ir_c=ir_c,
                           ir_f=ir_f,
                           ir_h=h,
                           filename_ident=filename_ident,
                           environment=env)

    fdimport = FunctionDefImporter(fctx, ast_fd)
    fdimport.import_body()
    return ir_f

  def _create_const_global_env(self, f, parameter_bindings, target):
    """Helper to generate an environment for a global function.

    This is a helper for the very common case and will be wholly insufficient
    for advanced cases, including mutable global state, closures, etc.
    Globals from the module are considered immutable.
    """
    ir_h = self._ir_h
    try:
      code = f.__code__
      globals_dict = f.__globals__
      builtins_module = globals_dict["__builtins__"]
    except AttributeError:
      assert False, (
          "Function {} does not have required user-defined function attributes".
          format(f))

    # Locals resolver.
    # Note that co_varnames should include both parameter and local names.
    locals_resolver = LocalNameResolver(code.co_varnames)
    resolvers = (
        locals_resolver,
        ConstModuleNameResolver(globals_dict, as_dict=True),
        ConstModuleNameResolver(builtins_module),
    )
    env = Environment(config=self._config, ir_h=ir_h, name_resolvers=resolvers)

    # Bind parameters.
    for name, value in parameter_bindings:
      logging.debug("STORE PARAM: {} <- {}", name, value)
      locals_resolver.checked_resolve_name(name).store(env, value)
    return env

  def _resolve_signature_annotation(self, target: Target, annot):
    ir_h = self._ir_h
    if annot is inspect.Signature.empty:
      return ir_h.basicpy_UnknownType

    # TODO: Do something real here once we need more than the primitive types.
    if annot is int:
      return target.impl_int_type
    elif annot is float:
      return target.impl_float_type
    elif annot is bool:
      return ir_h.basicpy_BoolType
    elif annot is str:
      return ir_h.basicpy_StrType
    else:
      return ir_h.basicpy_UnknownType


################################################################################
# Support
################################################################################


# TODO: Remove this hack in favor of a helper function that combines
# multiple dialect helpers so that we don't need to deal with the sharp
# edge of initializing multiple native base classes.
class AllDialectHelper(Numpy.DialectHelper, ScfDialectHelper):

  def __init__(self, *args, **kwargs):
    Numpy.DialectHelper.__init__(self, *args, **kwargs)
    ScfDialectHelper.__init__(self, *args, **kwargs)
