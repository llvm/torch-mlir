add_library(NPCOMPPythonCommon
  MlirInit.cpp
  MlirIr.cpp
  MlirPass.cpp
  NpcompDialect.cpp
  PybindUtils.cpp
  CoreDialects.cpp
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
