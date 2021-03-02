import numpy as np

from npcomp.compiler.numpy import test_config
from npcomp.compiler.numpy.backend import refjit
from npcomp.compiler.numpy.frontend import *
from npcomp.compiler.numpy.target import *


def compile_function(f):
  fe = ImportFrontend(config=test_config.create_test_config(
      target_factory=GenericTarget32))
  fe.import_global_function(f)
  compiler = refjit.CompilerBackend()
  vm_blob = compiler.compile(fe.ir_module)
  loaded_m = compiler.load(vm_blob)
  return loaded_m[f.__name__]


global_data = (np.zeros((2, 3)) + [1.0, 2.0, 3.0] * np.reshape([1.0, 2.0],
                                                               (2, 1)))

a = np.asarray([1.0, 2.0], dtype=np.float32)
b = np.asarray([3.0, 4.0], dtype=np.float32)


@compile_function
def global_add():
  return np.add(a, np.add(b, a))


assert global_add.__isnpcomp__

# CHECK: GLOBAL_ADD: [5. 8.]
result = global_add()
print("GLOBAL_ADD:", result)
