add_mlir_dialect_library(NPCOMPRefBackendDialect
  RefBackendDialect.cpp
  RefBackendOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/RefBackend

  DEPENDS
  MLIRRefBackendOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
