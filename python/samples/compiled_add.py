# Run full pipeline with:
#   -npcomp-cpa-type-inference -numpy-public-functions-to-tensor -convert-numpy-to-tcf -canonicalize

import numpy as np
from npcomp.compiler import test_config

import_global = test_config.create_import_dump_decorator()

global_data = (np.zeros((2, 3)) + [1.0, 2.0, 3.0] * np.reshape([1.0, 2.0],
                                                               (2, 1)))

a = np.asarray([1.0, 2.0], dtype=np.float32)
b = np.asarray([3.0, 4.0], dtype=np.float32)


@import_global
def global_add():
  return np.add(a, np.add(b, a))
