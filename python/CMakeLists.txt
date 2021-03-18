################################################################################
# Resources that must be packaged into the python tree
################################################################################

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/npcomp/compiler/backend/refjit_resources")
add_custom_target(NPCOMPPythonResources)
add_custom_command(
  TARGET NPCOMPPythonResources
  COMMAND ${CMAKE_COMMAND} -E copy
        # TODO: Make the runtime library work for windows.
        ${CMAKE_BINARY_DIR}/lib/libNPCOMPCompilerRuntimeShlib${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_CURRENT_BINARY_DIR}/npcomp/compiler/generic/backend/libNPCOMPCompilerRuntimeShlib${CMAKE_SHARED_LIBRARY_SUFFIX}
)
add_dependencies(NPCOMPPythonResources
  NPCOMPCompilerRuntimeShlib
  )


################################################################################
# Manage python source files
################################################################################
npcomp_python_create_symlinks(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

################################################################################
# Native extensions
################################################################################

# Normally on unix-like platforms, extensions are built as "MODULE" libraries
# and do not explicitly link to the python shared object. This allows for
# come greater deployment flexibility since the extension will bind to
# symbols in the python interpreter on load. However, it also keeps the
# linker from erroring on undefined symbols, leaving this to (usually obtuse)
# runtime errors. Building in "SHARED" mode with an explicit link to the
# python libraries allows us to build with the expectation of no undefined
# symbols, which is better for development.
# TODO(laurenzo): Windows requires linking against the PYTHON_LIBRARIES
# TODO(laurenzo): OSX requires allowing undefined (-undefined dynamic_lookup)
if(NPCOMP_PYTHON_BINDINGS_VERSION_LOCKED)
  set(NPCOMP_PYEXT_LINK_MODE SHARED)
  set(NPCOMP_PYEXT_LIBADD ${Python3_LIBRARIES})
else()
  set(NPCOMP_PYEXT_LINK_MODE MODULE)
  set(NPCOMP_PYEXT_LIBADD)
endif()

if(NPCOMP_ENABLE_REFJIT)
  list(APPEND NPCOMP_PYEXT_LIBADD
    NPCOMPBackendRefJITPythonModule
  )
endif()

add_library(NPCOMPNativePyExt ${NPCOMP_PYEXT_LINK_MODE}
  NpcompModule.cpp
)

install(TARGETS NPCOMPNativePyExt
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

install(DIRECTORY npcomp
DESTINATION python COMPONENT python)

set_target_properties(NPCOMPNativePyExt PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(NPCOMPNativePyExt PROPERTIES OUTPUT_NAME _npcomp)
set_target_properties(NPCOMPNativePyExt PROPERTIES PREFIX
                      "${PYTHON_MODULE_PREFIX}")
set_target_properties(NPCOMPNativePyExt PROPERTIES SUFFIX
                      "${PYTHON_MODULE_EXTENSION}")

# pybind requires binding code to be compiled with -fvisibility=hidden
# Better code can be generated if the entire project compiles that way, but
# that is not enforced here.
set_target_properties(NPCOMPNativePyExt PROPERTIES CXX_VISIBILITY_PRESET "hidden")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(NPCOMPNativePyExt
  PRIVATE
    # Transitive dep on the shared library links most things from there.
    NPCOMP

    NPCOMPInitAll
    ${NPCOMP_PYEXT_LIBADD}
  )
npcomp_python_target_compile_options(NPCOMPNativePyExt)

mlir_check_all_link_libraries(NPCOMPNativePyExt)

# Order dependent: Built artifacts add dependencies to the above targets.
add_subdirectory(npcomp/dialects)
