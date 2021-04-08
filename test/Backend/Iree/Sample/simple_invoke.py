# RUN: %PYTHON %s
# TODO: Numpy compiler has bitrotted.
# XFAIL: *

from npcomp.compiler.numpy.backend import iree
from npcomp.compiler.numpy.frontend import *
from npcomp.compiler.numpy import test_config
from npcomp.compiler.numpy.target import *
from npcomp.compiler.utils import logging

# TODO: This should all exist in a high level API somewhere.
from _npcomp import mlir


logging.enable()


def compile_function(f):
  fe = ImportFrontend(config=test_config.create_test_config(
    target_factory=GenericTarget32))
  fe.import_global_function(f)
  compiler = iree.CompilerBackend()
  vm_blob = compiler.compile(fe.ir_module)
  loaded_m = compiler.load(vm_blob)
  return loaded_m[f.__name__]


@compile_function
def int_add(a: int, b: int):
  return a + b

result = int_add(5, 6)
assert result == 11


@compile_function
def simple_control_flow(a: int, b: int):
  return (a * b) and (a - b)

assert simple_control_flow(5, 6) == -1
assert simple_control_flow(-1, 0) == 0
