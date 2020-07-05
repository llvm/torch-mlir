################################################################################
# NPCOMPBackendIREEPythonModule
################################################################################

include(NpcompPython)
include(NpcompIREEBackend)

set(PYBIND_SOURCES
  PythonModule.cpp
)
add_library(NPCOMPBackendIREEPythonModule
  ${PYBIND_SOURCES}
)

target_link_libraries(NPCOMPBackendIREEPythonModule
  NPCOMPPythonCommon
  iree_compiler_Dialect_Flow_Transforms_Transforms
  iree_compiler_Dialect_HAL_Transforms_Transforms
  iree_compiler_Dialect_VM_Transforms_Transforms
)

npcomp_python_target_compile_options(NPCOMPBackendIREEPythonModule)
npcomp_iree_target_compile_options(NPCOMPBackendIREEPythonModule)
