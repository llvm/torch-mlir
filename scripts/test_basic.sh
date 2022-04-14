./build/bin/torch-mlir-opt < test/Conversion/TorchToMhlo/basic.mlir -convert-torch-to-mhlo -split-input-file -verify-diagnostics
