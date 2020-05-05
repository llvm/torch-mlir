add_mlir_dialect_library(NPCOMPBasicpyDialect
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Basicpy

  DEPENDS
  MLIRBasicpyOpsIncGen
)

target_link_libraries(NPCOMPBasicpyDialect PUBLIC MLIRIR)
