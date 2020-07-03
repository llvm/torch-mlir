# RUN: %PYTHON %s | npcomp-opt -split-input-file -basicpy-cpa-type-inference | FileCheck %s --dump-input=fail

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
@import_global
def global_add():
  return np.add(a, b)
