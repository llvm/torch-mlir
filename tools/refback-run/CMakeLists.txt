# refback-run is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

get_property(dialect_libs GLOBAL PROPERTY NPCOMP_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY NPCOMP_CONVERSION_LIBS)

add_npcomp_executable(refback-run
  refback-run.cpp
  )

llvm_update_compile_flags(refback-run)
target_link_libraries(refback-run PRIVATE
  NPCOMPCAPI
  NPCOMPInitAll
  MLIRAnalysis
  MLIRIR
  MLIRJitRunner
  MLIRParser
  MLIRSupport
  NPCOMPInitAll
  NPCOMPRefBackendJITHelpers

  # TODO: Remove these in favor of interface deps.
  ${conversion_libs}
  ${dialect_libs}
)
add_dependencies(refback-run
  NPCOMPCompilerRuntimeShlib
  )
