################################################################################
# NPCOMPPythonCommon
################################################################################

include(NpcompPython)

set(PYBIND_SOURCES
  MlirInit.cpp
  MlirIr.cpp
  MlirPass.cpp
  NpcompDialect.cpp
  PybindUtils.cpp
  CoreDialects.cpp
)

add_library(NPCOMPPythonCommon
  ${PYBIND_SOURCES}
)

target_link_libraries(NPCOMPPythonCommon
  pybind11::module

  NPCOMPInitAll

  # Core dialects
  MLIRSCF

  # Upstream depends
  LLVMSupport
  MLIRDialect
  MLIREDSC
  MLIREDSCInterface
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRTransforms
)

npcomp_python_target_compile_options(NPCOMPPythonCommon)
