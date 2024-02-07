# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(TORCH_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir")

# We vendor our own MLIR instance in the `torch_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.")

# ################################################################################
# # Sources
# ################################################################################

declare_mlir_python_sources(TorchMLIRPythonSources.TopLevel
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonTorchExtensionsSources
  SOURCES
    torchscript.py
    _dynamo_fx_importer.py
    compiler_utils.py
    dynamo.py
    _version.py
)

################################################################################
# Lazy Tensor Core
################################################################################

# Reference backend has a separate check for TORCH_MLIR_ENABLE_LTC, since it
# generates a dummy Python library when disabled.
if(NOT TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS)
  add_subdirectory(torch_mlir/csrc/reference_lazy_backend)
endif()

################################################################################
# Optionally handle JIT IR importer.
################################################################################

if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  add_subdirectory(torch_mlir/jit_ir_importer)
  add_subdirectory(torch_mlir/csrc/jit_ir_importer)
  add_subdirectory(torch_mlir_e2e_test)
endif()

################################################################################
# Custom op example
# Required for running the update_torch_ods.sh and update_abstract_interp_lib.sh
# scripts.
################################################################################

# add_subdirectory(torch_mlir/_torch_mlir_custom_op_example)

# TODO: Find a cleaner way to do this.
# Can we build the JIT IR importer with `declare_mlir_python_extension`?
# Then it would "just work".
if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  add_dependencies(TorchMLIRPythonTorchExtensionsSources 
    TorchMLIRJITIRImporter
    TorchMLIRJITIRImporterPybind
    TorchMLIRE2ETestPythonModules
  )
endif()

if(TORCH_MLIR_ENABLE_LTC)
  # Add Torch-MLIR LTC backend as dependency
  add_dependencies(TorchMLIRPythonTorchExtensionsSources 
    torch_mlir_ltc_backend
    reference_lazy_backend
  )
endif()

add_subdirectory(test)
