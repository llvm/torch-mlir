## Declare the sources of the Python module.

declare_mlir_python_sources(TorchMLIRPythonSources.TorchScriptE2ETest
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES_GLOB
    dialects/torch/e2e_test/torchscript/*.py
)
