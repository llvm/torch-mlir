# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

import numpy as np
from npcomp.compiler import test_config

import_global = test_config.create_import_dump_decorator()

global_data = (np.zeros((2, 3)) + [1.0, 2.0, 3.0] * np.reshape([1.0, 2.0],
                                                               (2, 1)))


# CHECK-LABEL: func @global_array_to_const
@import_global
def global_array_to_const():
  # CHECK: %[[CST:.*]] =  constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  # CHECK: numpy.create_array_from_tensor %[[CST]] : (tensor<2x3xf64>) -> !numpy.ndarray<[2,3]:f64>
  local_data = global_data
  return local_data
