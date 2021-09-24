add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)

################################################################################
# Setup the initialization target.
# This includes conditional dependencies based on whether features are enabled.
################################################################################

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(npcomp_dialect_libs GLOBAL PROPERTY NPCOMP_DIALECT_LIBS)
get_property(npcomp_conversion_libs GLOBAL PROPERTY NPCOMP_CONVERSION_LIBS)

message(STATUS "NPCOMP Dialect libs: ${npcomp_dialect_libs}")
message(STATUS "NPCOMP Conversion libs: ${npcomp_conversion_libs}")

add_npcomp_library(NPCOMPInitAll
  InitAll.cpp

  LINK_LIBS

  PUBLIC
  # Local depends
  TorchMLIRTorchDialect
  NPCOMPTorchConversionDialect
  NPCOMPConversionPasses

  # TODO: We shouldn't need npcomp_conversion_libs here, but we have
  # some dialect transform libraries accumulating into that property.
  ${npcomp_conversion_libs}
  ${npcomp_dialect_libs}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
)
