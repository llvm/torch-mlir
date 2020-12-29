################################################################################
# NPCOMPBackendRefJITPythonModule
################################################################################

include(NpcompPython)

set(PYBIND_SOURCES
  PythonModule.cpp
)
add_library(NPCOMPBackendRefJITPythonModule
  ${PYBIND_SOURCES}
)

target_link_libraries(NPCOMPBackendRefJITPythonModule
  pybind11::module
  MLIRExecutionEngine
  MLIRTargetLLVMIR

  NPCOMPRefBackendJITHelpers
)

npcomp_python_target_compile_options(NPCOMPBackendRefJITPythonModule)

add_dependencies(NPCOMPBackendRefJITPythonModule
  NPCOMPPythonResources
)
