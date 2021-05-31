# Building libmlir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  message(WARNING "Not building NPCOMP dylib because PIC is disabled")
  return()
endif()

# Building libmlir-cpp.so may not work on MSVC
if (MSVC)
  message(WARNING "Not building NPCOMP dylib because not yet supported on MSVC")
  return()
endif()

if(NOT NPCOMP_BUILD_NPCOMP_DYLIB)
  message(WARNING "Not building NPCOMP dylib (not NPCOMP_BUILD_NPCOMP_DYLIB): Fully static builds not yet supported")
  return()
endif()

get_property(npcomp_libs GLOBAL PROPERTY NPCOMP_STATIC_LIBS)
list(REMOVE_DUPLICATES npcomp_libs)

# Populate _OBJECTS and _DEPS as necessary per platform.
foreach (lib ${npcomp_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # Add transitive deps explcitly since otherwise, there would just be
  # objects.
  list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach()

# Link to MLIR dylib if requested.
if(NPCOMP_LINK_MLIR_DYLIB)
  list(PREPEND _DEPS MLIR)
endif()

# Note: Does not use add_npcomp_library, which is used for things that go
# *into* the libNPCOMP.so. This is building the shared library, so use
# a higher-level rule.
llvm_add_library(
  NPCOMP
  SHARED
  npcomp-shlib.cpp
  ${_OBJECTS}
  LINK_LIBS PUBLIC
  # Public dependencies on the MLIR public API.
  MLIRPublicAPI
  ${_DEPS}
)
target_link_libraries(NPCOMP PRIVATE ${LLVM_PTHREAD_LIB})
