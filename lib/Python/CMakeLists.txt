################################################################################
# NPCOMPPythonCommon
################################################################################

include(AddLLVM)
include(NpcompPython)

# TODO: This should not be wired in at such a low/unconditional level.
# It is done here to be kept with the other LLVM initialization until a better
# place can be found for it.
set(ExtraInit_LIBADD)
if(NPCOMP_ENABLE_REFJIT)
  llvm_map_components_to_libnames(refjit_llvm_libs
    nativecodegen
  )
  message(STATUS "Including LLVM libs for RefJit: ${refjit_llvm_libs}")
  list(APPEND ExtraInit_LIBADD
    ${refjit_llvm_libs})
endif()

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
  MLIRDialect
  MLIREDSC
  MLIREDSCInterface
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRTransforms
)

npcomp_python_target_compile_options(NPCOMPPythonCommon)

mlir_check_all_link_libraries(NPCOMPPythonCommon)
