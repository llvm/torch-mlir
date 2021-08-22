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

from ..utils import logging
from .importer import *
from .interfaces import *
from .name_resolver_base import *
from .value_coder_base import *
from .target import *

from ..utils.mlir_utils import *

from ... import ir as _ir

__all__ = [
    "ImportFrontend",
]


class ImportFrontend:
  """Frontend for importing various entities into a Module."""
  __slots__ = [
      "_ir_module",
      "_config",
      "_ic",
  ]

  def __init__(self,
               *,
               config: Configuration,
               ir_context: Optional[_ir.Context] = None):
    super().__init__()
    ic = self._ic = ImportContext(ir_context)
    self._ic.module = _ir.Module.create(loc=ic.loc)
    self._config = config

  @property
  def ir_context(self) -> _ir.Context:
    return self._ic.context

  @property
  def ir_module(self) -> _ir.Module:
    return self._ic.module

  def import_global_function(self, f):
    """Imports a global function.

    This facility is not general and does not allow customization of the
    containing environment, method import, etc.

    Most errors are emitted via the MLIR context's diagnostic infrastructure,
    but errors related to extracting source, etc are raised directly.

    Args:
      f: The python callable.
    """
    ic = self._ic
    target = self._config.target_factory(ic)
    filename = inspect.getsourcefile(f)
    source_lines, start_lineno = inspect.getsourcelines(f)
    source = "".join(source_lines)
    source = textwrap.dedent(source)
    ast_root = ast.parse(source, filename=filename)
    ast.increment_lineno(ast_root, start_lineno - 1)
    ast_fd = ast_root.body[0]

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
    ir_f_type = _ir.FunctionType.get(f_input_types, [f_return_type],
                                     context=ic.context)

    ic.set_file_line_col(filename, ast_fd.lineno, ast_fd.col_offset)
    ic.insert_end_of_block(ic.module.body)
    ir_f, entry_block = ic.FuncOp(ast_fd.name,
                                  ir_f_type,
                                  create_entry_block=True)
    ic.insert_end_of_block(entry_block)
    env = self._create_const_global_env(f,
                                        parameter_bindings=zip(
                                            f_params.keys(),
                                            entry_block.arguments),
                                        target=target)
    fctx = FunctionContext(ic=ic, ir_f=ir_f, filename=filename, environment=env)

    fdimport = FunctionDefImporter(fctx, ast_fd)
    fdimport.import_body()
    return ir_f

  def _create_const_global_env(self, f, parameter_bindings, target):
    """Helper to generate an environment for a global function.

    This is a helper for the very common case and will be wholly insufficient
    for advanced cases, including mutable global state, closures, etc.
    Globals from the module are considered immutable.
    """
    ic = self._ic
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
    env = Environment(config=self._config, ic=ic, name_resolvers=resolvers)

    # Bind parameters.
    for name, value in parameter_bindings:
      logging.debug("STORE PARAM: {} <- {}", name, value)
      locals_resolver.checked_resolve_name(name).store(env, value)
    return env

  def _resolve_signature_annotation(self, target: Target, annot):
    ic = self._ic
    if annot is inspect.Signature.empty:
      return ic.unknown_type

    # TODO: Do something real here once we need more than the primitive types.
    if annot is int:
      return target.impl_int_type
    elif annot is float:
      return target.impl_float_type
    elif annot is bool:
      return ic.bool_type
    elif annot is str:
      return ic.str_type
    else:
      return ic.unknown_type
