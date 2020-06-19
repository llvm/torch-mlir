# RUN: %PYTHON %s

from npcomp.compiler.frontend import *
from npcomp.compiler.target import *

# TODO: This should all exist in a high level API somewhere.
from _npcomp import mlir
from _npcomp.backend import iree as ireec

from pyiree import rt


def compile_function(f):
  fe = ImportFrontend(target_factory=GenericTarget32)
  ir_f = fe.import_global_function(f)

  input_m = fe.ir_module
  # For easier debugging, split into to pass manager invocations.
  pm = mlir.passes.PassManager(input_m.context)
  # TOOD: Have an API for this
  pm.addPassPipelines(
    "basicpy-type-inference", "convert-basicpy-to-std", "canonicalize")
  pm.run(input_m)
  print("INPUT MODULE:")
  print(input_m.to_asm())

  # Main IREE compiler.
  pm = mlir.passes.PassManager(input_m.context)
  ireec.build_flow_transform_pass_pipeline(pm)
  ireec.build_hal_transform_pass_pipeline(pm)
  ireec.build_vm_transform_pass_pipeline(pm)
  pm.run(input_m)
  print("VM MODULE:")
  print(input_m.to_asm())

  # Translate to VM bytecode flatbuffer.
  vm_blob = ireec.translate_to_vm_bytecode(input_m)
  print("VM BLOB: len =", len(vm_blob))
  return vm_blob


def int_add(a: int, b: int):
  return a + b

vm_blob = compile_function(int_add)
m = rt.VmModule.from_flatbuffer(vm_blob)
config = rt.Config("vmla")
ctx = rt.SystemContext(config=config)
ctx.add_module(m)

f = ctx.modules.module.int_add
print(f(5, 6))
