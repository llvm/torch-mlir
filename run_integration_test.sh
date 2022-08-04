#!/usr/bin/env bash

export PYTHONPATH="$(pwd)/build/tools/torch-mlir/python_packages/torch_mlir"


# refbackend e2e tests
python -m e2e_testing.torchscript.main --config=refbackend -v

# eagermode backend e2e tests
python -m e2e_testing.torchscript.main --config=eager_mode -v

# tosa backend e2e tests
python -m e2e_testing.torchscript.main --config=tosa -v

# ltc backend e2e tests
python -m e2e_testing.torchscript.main --config=lazy_tensor_core -v
