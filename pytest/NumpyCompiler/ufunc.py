# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

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
  # CHECK-DAG: %[[CST_A_TENSOR:.*]] = constant dense<[1.000000e+00, 2.000000e+00]>
  # CHECK-DAG: %[[CST_B_TENSOR:.*]] = constant dense<[3.000000e+00, 4.000000e+00]>
  # CHECK-DAG: %[[A_ARRAY:.*]] = numpy.create_array_from_tensor %[[CST_A_TENSOR]]
  # CHECK-DAG: %[[B_ARRAY:.*]] = numpy.create_array_from_tensor %[[CST_B_TENSOR]]
  # CHECK-DAG: %[[A:.*]] = numpy.copy_to_tensor %[[A_ARRAY]]
  # CHECK-DAG: %[[B:.*]] = numpy.copy_to_tensor %[[B_ARRAY]]
  # CHECK: %[[R_TENSOR:.*]] = numpy.builtin_ufunc_call<"numpy.add"> (%[[A]], %[[B]]) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*x!basicpy.UnknownType>
  # CHECK: numpy.create_array_from_tensor %[[R_TENSOR]] : (tensor<*x!basicpy.UnknownType>) -> !numpy.ndarray<?>
  return np.add(a, b)


@import_global(expect_error="ufunc call does not currently support keyword args"
              )
def keywords_not_supported():
  return np.add(a, b, out=b)


@import_global(expect_error="ufunc numpy.add expected 2 inputs but got 1")
def mismatched_arg_count():
  return np.add(a)
