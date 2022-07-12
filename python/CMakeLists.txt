include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(TORCH_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir")


# We vendor our own MLIR instance in the `torch_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TorchMLIRPythonSources)
declare_mlir_python_sources(TorchMLIRPythonExtensions)

declare_mlir_python_sources(TorchMLIRPythonSources.TopLevel
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    __init__.py
    compiler_utils.py
)

declare_mlir_python_sources(TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TorchBinding.td
  SOURCES dialects/torch/__init__.py
  DIALECT_NAME torch
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(TorchMLIRPythonExtensions.Main
  MODULE_NAME _torchMlir
  ADD_TO_PARENT TorchMLIRPythonExtensions
  SOURCES
    TorchMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    TorchMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Lazy Tensor Core
################################################################################

add_subdirectory(torch_mlir/csrc/base_lazy_backend)
add_subdirectory(torch_mlir/csrc/reference_lazy_backend)

################################################################################
# Optionally handle JIT IR importer.
################################################################################

option(LIBTORCH_SRC_BUILD "Build libtorch from source" OFF)
option(LIBTORCH_CACHE "Cache libtorch downloads" OFF)
option(LIBTORCH_VARIANT "libtorch variant [shared|static]-[cxx11]" "libtorch-shared")

if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  # if LIBTORCH_SRC_BUILD=OFF (default) we link against a binary libtorch :
  #   if LIBTORCH_CACHE=OFF (default) the build downloads the latest released version of libtorch
  #   if LIBTORCH_CACHE=ON the build uses an available libtorch/ in the root directory.
  #   In either case if there is no libtorch/ the latest released version for the platform is downloaded
  #   A developer could unpack their own libtorch/ and it would be respected
  # if LIBTORCH_SRC_BUILD=ON we build libtorch/pytorch and link against it
  #   if LIBTORCH_VARIANT=*cxx11abi*  we build with the new CXX11 ABI similar to upstream pytorch/builder
  #   if LIBTORCH_VARIANT=*shared*   (default) we build a shared library (and pytorch .whl)
  #     else if LIBTORCH_VARIANT=*static*   we build a static libtorch
  set(ENV{LIBTORCH_SRC_BUILD} ${LIBTORCH_SRC_BUILD})
  set(ENV{LIBTORCH_CACHE} ${LIBTORCH_CACHE})
  set(ENV{LIBTORCH_VARIANT} ${LIBTORCH_VARIANT})
  set(ENV{CMAKE_OSX_ARCHITECTURES} ${CMAKE_OSX_ARCHITECTURES})
  set(ENV{CMAKE_C_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER_LAUNCHER})
  set(ENV{CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_CXX_COMPILER_LAUNCHER})
  execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../build_tools/build_libtorch.sh
    RESULT_VARIABLE _result
  )
  if(_result)
    message(FATAL_ERROR "Failed to run `build_libtorch.sh`")
  endif()
  set(TORCH_INSTALL_PREFIX "libtorch")
  add_subdirectory(torch_mlir/dialects/torch/importer/jit_ir)
  add_subdirectory(torch_mlir_e2e_test)
endif()

################################################################################
# Eager mode
################################################################################

add_subdirectory(torch_mlir/eager_mode)

################################################################################
# Custom op example
# Required for running the update_torch_ods.sh and update_shape_lib.sh scripts.
################################################################################

# TODO: renable once it build on macOS Intel / M1
#add_subdirectory(torch_mlir/_torch_mlir_custom_op_example)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies. We need ExecutionEngine and a bunch of passes
  # for the reference backend, but logically they can be separate. But seemingly
  # the only way to handle that is to create a separate mlir python package
  # tree, which seems excessive.
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.AllPassesRegistration
  MLIRPythonExtension.ExecutionEngine
  TorchMLIRPythonSources
  TorchMLIRPythonExtensions
)

add_mlir_python_common_capi_library(TorchMLIRAggregateCAPI
  INSTALL_COMPONENT TorchMLIRPythonModules
  INSTALL_DESTINATION python_packages/torch_mlir/torch_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(TorchMLIRPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir"
  INSTALL_PREFIX "python_packages/torch_mlir/torch_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    TorchMLIRAggregateCAPI
  )

# TODO: Find a cleaner way to do this.
# Can we build the JIT IR importer with `declare_mlir_python_extension`?
# Then it would "just work".
if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  add_dependencies(TorchMLIRPythonModules TorchMLIRJITIRImporter)
  # Build the E2E Tests (which depend on the JIT IR importer now).
  add_dependencies(TorchMLIRPythonModules TorchMLIRE2ETestPythonModules)
endif()

# TODO: Add after macOS builds are fixed
#add_dependencies(TorchMLIRPythonModules torch_mlir_custom_op_example)

# Add Torch-MLIR LTC backend as dependency
add_dependencies(TorchMLIRPythonModules torch_mlir_ltc_backend)
add_dependencies(TorchMLIRPythonModules reference_lazy_backend)

add_subdirectory(test)
