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
  "${Python3_LIBRARIES}"
  pybind11::module
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport

  NPCOMPRefBackendJITHelpers
)

npcomp_python_target_compile_options(NPCOMPBackendRefJITPythonModule)

add_dependencies(NPCOMPBackendRefJITPythonModule
  NPCOMPPythonResources
)
