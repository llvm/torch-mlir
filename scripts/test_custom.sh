./build/bin/torch-mlir-opt < test/Conversion/TorchToMhlo/custom.mlir -convert-torch-to-mhlo -split-input-file -verify-diagnostics
