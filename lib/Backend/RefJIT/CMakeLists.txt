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
  MLIRExecutionEngine
  MLIRTargetLLVMIR

  NPCOMPPythonCommon
  NPCOMPJITRuntime
)

npcomp_python_target_compile_options(NPCOMPBackendRefJITPythonModule)
