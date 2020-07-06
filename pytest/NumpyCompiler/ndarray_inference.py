# RUN: %PYTHON %s | npcomp-opt -split-input-file -npcomp-cpa-type-inference | FileCheck %s --dump-input=fail

import numpy as np
from npcomp.compiler import test_config
from npcomp.compiler.frontend import EmittedError

import_global = test_config.create_import_dump_decorator()

global_data = (np.zeros((2, 3)) + [1.0, 2.0, 3.0] * np.reshape([1.0, 2.0],
                                                               (2, 1)))

a = np.asarray([1.0, 2.0])
b = np.asarray([3.0, 4.0])


# Test the basic flow of invoking a ufunc call with constants captured from
# a global using explicit function syntax (np.add(a, b)).
# CHECK-LABEL: func @global_add
# CHECK-SAME: -> !numpy.ndarray<*:f64>
@import_global
def global_add():
  # CHECK-NOT: UnknownType
  # CHECK: numpy.builtin_ufunc_call<"numpy.add"> ({{.*}}, {{.*}}) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  # CHECK-NOT: UnknownType
  return np.add(a, b)
