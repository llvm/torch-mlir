################################################################################
# NPCOMPBackendIREEPythonModule
################################################################################

set(PYBIND_SOURCES
  PythonModule.cpp
)
set_source_files_properties(
  ${PYBIND_SOURCES}
  PROPERTIES COMPILE_FLAGS
    "-frtti -fexceptions")

add_library(NPCOMPBackendIREEPythonModule
  ${PYBIND_SOURCES}
)

target_link_libraries(NPCOMPBackendIREEPythonModule
  NPCOMPPythonCommon
  iree_compiler_Dialect_Flow_Transforms_Transforms
  iree_compiler_Dialect_HAL_Transforms_Transforms
  iree_compiler_Dialect_VM_Transforms_Transforms
)
