# Appease LLVM check that all sources are covered by a target.
# This doesn't seem to play well with having multiple targets
# in a single directory.
set(LLVM_OPTIONAL_SOURCES
  Runtime.cpp
  CompilerRuntime.cpp
)

# The library that users link against, defining basic interactions with an
# refbackrt module and the relevant data structures.
add_npcomp_library(NPCOMPRuntime
  Runtime.cpp
)

# If we aren't linking to MLIR dylib, link LLVMSupport.
if(NOT NPCOMP_LINK_MLIR_DYLIB)
  target_link_libraries(NPCOMPRuntime PRIVATE LLVMSupport)
endif()

mlir_check_all_link_libraries(NPCOMPRuntime)

# The library that defines the symbols that the compiler emits references
# to.
# Note: is uses some of the same facilities that the user API depends on,
# we use a linker script to ensure that the shared library only exposes the
# symbols the compiler needs.
#
# This is currently done as a shared library to make it suitable for being
# loaded by mlir::ExecutionEngine. In e.g. an embedded scenario, we would
# need to create a static library and link that into the binary.
add_npcomp_library(NPCOMPCompilerRuntimeShlib
  SHARED
  CompilerRuntime.cpp

  EXCLUDE_FROM_LIBNPCOMP
)
target_link_libraries(NPCOMPCompilerRuntimeShlib PRIVATE NPCOMPRuntime)
if (UNIX AND NOT APPLE)
set_target_properties(NPCOMPCompilerRuntimeShlib PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/unix_version.script")
endif()
