# torch-mlir

The Torch-MLIR project aims to provide first class support from the [Pytorch](https://pytorch.org) ecosystem to the MLIR ecosystem.

> This project is participating in the LLVM Incubator process: as such, it is
not part of any official LLVM release.  While incubation status is not
necessarily a reflection of the completeness or stability of the code, it
does indicate that the project is not yet endorsed as a component of LLVM.

[Pytorch](https://pytorch.org)
An open source machine learning framework that accelerates the path from research prototyping to production deployment.

[MLIR](https://mlir.llvm.org)
The MLIR project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.

[Torch-MLIR](https://github.com/llvm/torch-mlir)
Multiple Vendors use MLIR as the middle layer mapping from Platform Frameworks like Pytorch, JAX, Tensorflow onto MLIR and then progressively lower down to their target hardware. We have seen half a dozen custom lowerings from PyTorch to MLIR. Having a canonical lowering from the Pytorch ecosystem to the MLIR ecosystem would provide much needed relief to Hardware Vendors to focus on their unique value rather than implementing another Pytorch frontend for MLIR. It would be similar to current hardware vendors adding LLVM target support instead of each one also implementing the Clang/C++ frontend.

## All the roads from PyTorch to Torch MLIR Dialect

We have few paths to lower down to the Torch MLIR Dialect.

![Torch Lowering Architectures](Torch-MLIR.png)

 - Torchscript
    This is the most tested path down to Torch MLIR Dialect.
 - TorchFX
	This provides a path to lower from TorchFX down to MLIR. This a functional prototype that we expect to mature as TorchFX matures
 - Lazy Tensor Core (Based on lazy-tensor-core [staging branch](https://github.com/pytorch/pytorch/tree/lazy_tensor_staging/lazy_tensor_core))
	This path provides the upcoming LTC path of capture. It is based of an unstable devel branch but is the closest way for you to adapt any existing torch_xla derivatives.
 - “ACAP”  - Deprecated torch_xla based capture Mentioned here for completeness.

## Examples
There are few examples of lowering down via path from PyTorch to MLIR and using the “mlir-cpu-runner” to target a CPU backend. Obviously this is just a starting point and you can import this project into your larger MLIR projects to continue lowering to target GPUs and other Accelerators.

## Project Communication

- `#torch-mlir` channel on the LLVM [Discord](https://discord.gg/xS7Z362)
- issues/PR's on this github repo
- [`torch-mlir` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/torch-mlir/41) of LLVM Discourse

## Repository Layout

The project follows the conventions of typical MLIR-based projects:

* `include/torch-mlir`, `lib` structure for C++ MLIR compiler dialects/passes.
* `test` for holding test code.
* `tools` for `torch-mlir-opt` and such.
* `python` top level directory for Python code

## Interactive Use

The `build_tools/write_env_file.sh` script will output a `.env`
file in the workspace folder with the correct PYTHONPATH set. This allows
tools like VSCode to work by default for debugging. This file can also be
manually `source`'d in a shell.

## Build Instructions

```shell
# From checkout directory.
git submodule init
git submodule update

# Use clang and lld to build (optional but recommended).
LLVM_VERSION=10
export CC=clang-$LLVM_VERSION
export CXX=clang++-$LLVM_VERSION
export LDFLAGS=-fuse-ld=$(which ld.lld-$LLVM_VERSION)

# Install PyTorch. We currently track and require the nighly build.
# If a usable PyTorch package is installed, the default cmake settings will
# enable the PyTorch frontend.
pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# Invoke CMake and build. This will also run all unit tests.
./build_tools/build_standalone.sh

# Run write_env_file.sh to emit a .env file with needed
# PYTHONPATH setup.
./build_tools/write_env_file.sh
source .env

```

## Demos

### TorchScript

A lot of the prior effort on torch-mlir has gone into the TorchScript compiler,
with a path to execution via linalg-on-tensors (a commonly used representation
of tensor computations in the MLIR community). Thus, this path has the most
extensive testing and functionality.

Running execution (end-to-end) tests:

```
# Run E2E TorchScript tests. These compile and run the TorchScript program
# through torch-mlir with a simplified linalg-on-tensors based backend we call
# RefBackend (more production-grade backends at this same abstraction layer
# exist in the MLIR community, such as IREE).
./tools/torchscript_e2e_test.sh --filter Conv2d --verbose
```

Standalone script:

```
# Run ResNet18 as a standalone script.
python examples/torchscript_resnet18_e2e.py
```

Jupyter notebook:
```
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH "$PYTHONPATH"
# Open in jupyter, and then navigate to
# `examples/resnet_inference.ipynb` and use the `torch-mlir` kernel to run.
jupyter notebook
```


### TorchFX

TODO


### Lazy Tensor Core

TODO

### Additional TorchScript end-to-end tests with heavy dependencies

Some of the Torchscript end-to-end tests require additional dependencies which
don't make sense to include as part of the default torch-mlir setup.
Additionally, these dependencies often don't work with the same HEAD PyTorch
version that torch-mlir builds against at the C++ level (the TorchScript
importer is written in C++)

We have a self-contained script that generates all the needed artifacts from a
self-contained virtual environment. It can be used like so:

```shell
# Build the virtual environment in the specified directory and generate the
# serialized test artifacts in the other specified directory.
# This command is safe to re-run if you have already built the virtual
# environment and just changed the tests.
build_tools/torchscript_e2e_heavydep_tests/generate_serialized_tests.sh \
  path/to/heavydep_venv \
  path/to/heavydep_serialized_tests

# Add the --serialized-test-dir flag to point at the directory containing the
# serialized tests. All other functionality is the same as the normal invocation
# of torchscript_e2e_test.sh, but the serialized tests will be available.
tools/torchscript_e2e_test.sh --serialized-test-dir=path/to/heavydep_serialized_tests
```

The tests use the same (pure-Python) test framework as the normal
torchscript_e2e_test.sh, but the tests are added in
`build_tools/torchscript_e2e_heavydep_tests` instead of
`frontends/pytorch/e2e_testing/torchscript`.

We rely critically on serialized TorchScript compatibility across PyTorch
versions to transport the tests + pure-Python compatibility of the `torch`
API, which has worked well so far.
