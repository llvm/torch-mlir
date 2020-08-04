################################################################################
# Resources that must be packaged into the python tree
################################################################################

add_custom_target(NPCOMPPythonResources)
add_custom_command(
    TARGET NPCOMPPythonResources
    COMMAND ${CMAKE_COMMAND} -E copy
        # TODO: Make the runtime library work for windows.
        ${CMAKE_BINARY_DIR}/lib/runtime/libNPCOMPCompilerRuntimeShlib.so
        ${CMAKE_CURRENT_BINARY_DIR}/npcomp/compiler/backend/refjit_resources/libNPCOMPCompilerRuntimeShlib.so
)
add_dependencies(NPCOMPPythonResources
  NPCOMPCompilerRuntimeShlib
  )


################################################################################
# Manage python source files
################################################################################
function (create_symlinks)
    # Do nothing if building in-source
    if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()

    foreach (path_file ${ARGN})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${folder}")

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${path_file}" link)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${path_file}" target)

        if (UNIX)
            set(command ln -s ${target} ${link})
        else()
            set(command cmd.exe /c mklink ${link} ${target})
        endif()

        execute_process(COMMAND ${command}
                        RESULT_VARIABLE result
                        ERROR_VARIABLE output)

        if (NOT ${result} EQUAL 0)
            message(FATAL_ERROR "Could not create symbolic link for: ${target} --> ${output}")
        endif()

    endforeach(path_file)
endfunction(create_symlinks)

file(GLOB_RECURSE python_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
create_symlinks(${python_files})

################################################################################
# Native extensions
################################################################################

include(NpcompPython)

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
set(NPCOMP_PYEXT_LINK_MODE SHARED)
set(NPCOMP_PYEXT_LIBADD ${PYTHON_LIBRARIES})

if(NPCOMP_ENABLE_IREE)
  list(APPEND NPCOMP_PYEXT_LIBADD NPCOMPBackendIREEPythonModule)
endif()

if(NPCOMP_ENABLE_REFJIT)
  list(APPEND NPCOMP_PYEXT_LIBADD
    NPCOMPBackendRefJITPythonModule
  )
endif()

# TODO(laurenzo): Add a config setting to control this.
# set(NPCOMP_PYEXT_LINK_MODE MODULE)
# set(NPCOMP_PYEXT_LIBADD "")

# When building the extension, distinguish between those sources that use
# pybind (and need rtti/exceptions) and those that only use LLVM/MLIR.
# Some of the low-level components do not support mixing RTTI modes and are
# compiled separately for now.
set(extension_target NPCOMPNativePyExt)
set(extension_llvm_sources
)
set(extension_pybind_sources
  NpcompModule.cpp
)
set_source_files_properties(
  ${extension_pybind_sources}
  PROPERTIES COMPILE_FLAGS "${extension_cflags}")
add_library(${extension_target} ${NPCOMP_PYEXT_LINK_MODE}
  ${extension_pybind_sources}
  ${extension_llvm_sources}
)

set_target_properties(${extension_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${extension_target} PROPERTIES OUTPUT_NAME _npcomp)
set_target_properties(${extension_target} PROPERTIES PREFIX
                      "${PYTHON_MODULE_PREFIX}")
set_target_properties(${extension_target} PROPERTIES SUFFIX
                      "${PYTHON_MODULE_EXTENSION}")

# pybind requires binding code to be compiled with -fvisibility=hidden
# Better code can be generated if the entire project compiles that way, but
# that is not enforced here. Instead, include a linker script that explicitly
# hides anything but the PyInit_* symbols, allowing gc to take place.
# TODO(laurenzo): Windows needs a .def file and different flags.
set_target_properties(${extension_target} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
set_target_properties(${extension_target} PROPERTIES LINK_FLAGS
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/unix_version.script")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# llvm_update_compile_flags(${extension_target})
target_link_libraries(${extension_target}
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    NPCOMPInitAll
    NPCOMPPythonCommon

    # Upstream depends
    LLVMSupport
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRDialect
    MLIREDSC
    MLIREDSCInterface
    MLIRIR
    MLIRSCFToStandard
    MLIRLLVMIR
    MLIRPass
    MLIRTargetLLVMIR
    MLIRTransforms

    ${NPCOMP_PYEXT_LIBADD}
)

npcomp_python_target_compile_options(${extension_target})
