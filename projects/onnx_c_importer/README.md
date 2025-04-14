# ONNX C Importer

This project provides a C++ implementation of the `onnx_importer.py`, which is
the canonical source. It is provided as sample code for anyone who wishes to
integrate it into their system. By design, it only depends on the ONNX API
and the MLIR C API via the `mlir-c` headers. As such, it should be easy to
build into any system that already has those things by adding the sources.
