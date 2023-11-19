#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

## Declare the sources of the Python module.

declare_mlir_python_sources(TorchMLIRPythonSources.JitIRImporter
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES_GLOB
    jit_ir_importer/*.py
)
