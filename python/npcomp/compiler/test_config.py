#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Various configuration helpers for testing."""

import ast
import functools

from . import logging
from .frontend import *
from .interfaces import *
from .partial_eval_base import *
from .target import *
from .value_coder_base import *
from .extensions import numpy as npc


def create_import_dump_decorator(*,
                                 target_factory: TargetFactory = GenericTarget64
                                ):
  config = create_test_config(target_factory=target_factory)
  logging.debug("Testing with config: {}", config)

  def do_import(f):
    fe = ImportFrontend(config=config)
    fe.import_global_function(f)
    print("// -----")
    print(fe.ir_module.to_asm())
    return f

  def decorator(*args, expect_error=None):
    if len(args) == 0:
      # Higher order decorator.
      return functools.partial(decorator, expect_error=expect_error)

    assert len(args) == 1
    try:
      return do_import(f=args[0])
    except EmittedError as e:
      if expect_error and e.message == expect_error:
        print("// EXPECTED_ERROR:", repr(e.message))
        pass
      elif expect_error:
        print("// MISMATCHED_ERROR:", repr(e.message))
        raise AssertionError("Expected error '{}' but got '{}'".format(
            expect_error, e.message))
      else:
        print("// UNEXPECTED_ERROR:", repr(e.message))
        raise e

  return decorator


def create_test_config(target_factory: TargetFactory = GenericTarget64):
  value_coder = ValueCoderChain([
      BuiltinsValueCoder(),
      npc.CreateNumpyValueCoder(),
  ])
  pe_hook = build_default_partial_eval_hook()

  # Populate numpy partial evaluators.
  npc.bind_ufuncs(npc.get_ufuncs_from_module(), pe_hook)

  if logging.debug_enabled:
    logging.debug("Partial eval mapping: {}", pe_hook)

  return Configuration(target_factory=target_factory,
                       value_coder=value_coder,
                       partial_eval_hook=pe_hook)


def build_default_partial_eval_hook() -> PartialEvalHook:
  pe = MappedPartialEvalHook()
  ### Modules
  pe.enable_getattr(for_type=ast.__class__)  # The module we use is arbitrary.

  ### Tuples
  # Enable attribute resolution on tuple, which includes namedtuple (which is
  # really what we want).
  pe.enable_getattr(for_type=tuple)

  ### Temp: resolve a function to a template call for testing
  import math
  pe.enable_template_call("__global$math.ceil", for_ref=math.ceil)
  pe.enable_template_call("__global$math.isclose", for_ref=math.isclose)
  return pe
