################################################################################
# NPCOMPPythonCommon
################################################################################

set(PYBIND_SOURCES
  MlirInit.cpp
  MlirIr.cpp
  MlirPass.cpp
  NpcompDialect.cpp
  PybindUtils.cpp
  CoreDialects.cpp
)
set_source_files_properties(
  ${PYBIND_SOURCES}
  PROPERTIES COMPILE_FLAGS
    "-frtti -fexceptions")

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
