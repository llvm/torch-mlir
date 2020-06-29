#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Various configuration helpers for testing."""

import ast

from . import logging
from .frontend import *
from .interfaces import *
from .partial_eval_base import *
from .target import *
from .value_coder_base import *
from .value_coder_numpy import *


def create_import_dump_decorator(*,
                                 target_factory: TargetFactory = GenericTarget64
                                ):
  config = create_test_config(target_factory=target_factory)
  logging.debug("Testing with config: {}", config)

  def decorator(f):
    fe = ImportFrontend(config=config)
    fe.import_global_function(f)
    print("// -----")
    print(fe.ir_module.to_asm())
    return f

  return decorator


def create_test_config(target_factory: TargetFactory = GenericTarget64):
  value_coder = ValueCoderChain([
      BuiltinsValueCoder(),
      CreateNumpyValueCoder(),
  ])
  pe_hook = build_default_partial_eval_hook()

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
