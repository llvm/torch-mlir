# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

import numpy as np

from npcomp.compiler.backend import refjit
from npcomp.compiler.frontend import *
from npcomp.compiler import logging
from npcomp.compiler import test_config
from npcomp.compiler.target import *

# TODO: This should all exist in a high level API somewhere.
from _npcomp import mlir

logging.enable()


def compile_function(f):
  fe = ImportFrontend(config=test_config.create_test_config(
      target_factory=GenericTarget32))
  fe.import_global_function(f)
  compiler = refjit.CompilerBackend()
  blob = compiler.compile(fe.ir_module)
  loaded_m = compiler.load(blob)
  return loaded_m[f.__name__]


global_data = (np.zeros((2, 3)) + [1.0, 2.0, 3.0] * np.reshape([1.0, 2.0],
                                                               (2, 1)))

a = np.asarray([1.0, 2.0], dtype=np.float32)
b = np.asarray([3.0, 4.0], dtype=np.float32)


@compile_function
def global_add():
  return np.add(a, np.add(b, a))


# Make sure we aren't accidentally invoking the python function :)
assert global_add.__isnpcomp__

# CHECK: GLOBAL_ADD: [5. 8.]
result = global_add()
print("GLOBAL_ADD:", result)
