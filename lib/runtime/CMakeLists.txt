# Appease LLVM check that all sources are covered by a target.
# This doesn't seem to play well with having multiple targets
# in a single directory.
set(LLVM_OPTIONAL_SOURCES
  Runtime.cpp
  CompilerRuntime.cpp
)

# The library that users link against, defining basic interactions with an
# npcomprt module and the relevant data structures.
add_mlir_library(NPCOMPRuntime
  Runtime.cpp

  LINK_LIBS
  MLIRExecutionEngine

  EXCLUDE_FROM_LIBMLIR
)

# The library that defines the symbols that the compiler emits references
# to.
# Note: is uses some of the same facilities that the user API depends on,
# we use a linker script to ensure that the shared library only exposes the
# symbols the compiler needs.
#
# This is currently done as a shared library to make it suitable for being
# loaded by mlir::ExecutionEngine. In e.g. an embedded scenario, we would
# need to create a static library and link that into the binary.
add_mlir_library(NPCOMPCompilerRuntimeShlib
  SHARED
  CompilerRuntime.cpp

  EXCLUDE_FROM_LIBMLIR
)
target_link_libraries(NPCOMPCompilerRuntimeShlib PRIVATE NPCOMPRuntime)
set_target_properties(NPCOMPCompilerRuntimeShlib PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/unix_version.script")

