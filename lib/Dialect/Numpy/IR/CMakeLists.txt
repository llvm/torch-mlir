add_mlir_dialect_library(NPCOMPNumpyDialect
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Numpy

  DEPENDS
  NPCOMPBasicpyDialect
  MLIRNumpyOpsIncGen
)

target_link_libraries(NPCOMPNumpyDialect PUBLIC MLIRIR)
