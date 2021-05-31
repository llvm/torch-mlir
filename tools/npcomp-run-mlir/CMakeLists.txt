# npcomp-run-mlir is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

get_property(dialect_libs GLOBAL PROPERTY NPCOMP_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY NPCOMP_CONVERSION_LIBS)

add_npcomp_executable(npcomp-run-mlir
  npcomp-run-mlir.cpp
  )

llvm_update_compile_flags(npcomp-run-mlir)
target_link_libraries(npcomp-run-mlir PRIVATE
  # Shared library deps first ensure we get most of what we need from libraries.
  NPCOMP

  NPCOMPCAPI
  MLIRAnalysis
  MLIRIR
  MLIRJitRunner
  MLIRParser
  MLIRSupport
  NPCOMPInitAll
  NPCOMPRefBackendJITHelpers
  ${conversion_libs}
  ${dialect_libs}
)
add_dependencies(npcomp-run-mlir
  NPCOMPCompilerRuntimeShlib
  )
