#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

## Declare the sources of the Python module.

declare_mlir_python_sources(TorchMLIRPythonSources.EagerMode
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES_GLOB eager_mode/*.py lazytensor/*.py
)
