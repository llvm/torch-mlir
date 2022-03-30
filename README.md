# The Torch-MLIR Project

The Torch-MLIR project aims to provide first class compiler support from the [PyTorch](https://pytorch.org) ecosystem to the MLIR ecosystem.

> This project is participating in the LLVM Incubator process: as such, it is
not part of any official LLVM release.  While incubation status is not
necessarily a reflection of the completeness or stability of the code, it
does indicate that the project is not yet endorsed as a component of LLVM.

[PyTorch](https://pytorch.org)
An open source machine learning framework that accelerates the path from research prototyping to production deployment.

[MLIR](https://mlir.llvm.org)
The MLIR project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.

[Torch-MLIR](https://github.com/llvm/torch-mlir)
Multiple Vendors use MLIR as the middle layer, mapping from platform frameworks like PyTorch, JAX, and TensorFlow into MLIR and then progressively lowering down to their target hardware. We have seen half a dozen custom lowerings from PyTorch to MLIR. Having canonical lowerings from the PyTorch ecosystem to the MLIR ecosystem would provide much needed relief to hardware vendors to focus on their unique value rather than implementing yet another PyTorch frontend for MLIR. The goal is to be similar to current hardware vendors adding LLVM target support instead of each one also implementing Clang / a C++ frontend.

## All the roads from PyTorch to Torch MLIR Dialect

We have few paths to lower down to the Torch MLIR Dialect.

![Torch Lowering Architectures](Torch-MLIR.png)

 - TorchScript
    This is the most tested path down to Torch MLIR Dialect, and the PyTorch ecosystem is converging on using TorchScript IR as a lingua franca.
 - LazyTensorCore (Based on the PyTorch [`lazy_tensor_staging` branch](https://github.com/pytorch/pytorch/tree/lazy_tensor_staging/lazy_tensor_core))
	This path provides the upcoming LTC path of capture. It is based of an unstable devel branch but is the closest way for you to adapt any existing `torch/xla` derivatives.

## Project Communication

- `#torch-mlir` channel on the LLVM [Discord](https://discord.gg/xS7Z362) - this is the most active communication channel
- Github issues [here](https://github.com/llvm/torch-mlir/issues)
- [`torch-mlir` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/torch-mlir/41) of LLVM Discourse

## Check out the code

```shell
git clone https://github.com/llvm/torch-mlir
cd torch-mlir
git submodule update --init
```

## Setup your Python VirtualEnvironment and Dependencies

```shell
python -m venv mlir_venv
source mlir_venv/bin/activate
# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
# Install latest PyTorch nightlies and build requirements.
python -m pip install -r requirements.txt
```

## Build
```shell
cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=`pwd` \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=`pwd`/externals/llvm-external-projects/torch-mlir-dialects \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm

# Additional quality of life CMake flags:
# Enable ccache:
#  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
# Enable LLD (links in seconds compared to minutes)
# -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
# Use --ld-path= instead of -fuse-ld=lld for clang > 13

# Build just torch-mlir (not all of LLVM)
cmake --build build --target tools/torch-mlir/all

# Run unit tests.
cmake --build build --target check-torch-mlir

# Build everything (including LLVM)
cmake --build build
```
## Demos

## Setup Python Environment
```shell
export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples
```

### TorchScript

Running execution (end-to-end) tests:

```shell
# Run E2E TorchScript tests. These compile and run the TorchScript program
# through torch-mlir with a simplified MLIR CPU backend we call RefBackend
python -m e2e_testing.torchscript.main --filter Conv2d --verbose
```

[Example IR](https://gist.github.com/silvasean/e74780f8a8a449339aac05c51e8b0caa) for a simple 1 layer MLP to show the compilation steps from TorchScript.

Standalone script to Convert a PyTorch ResNet18 model to MLIR and run it on the CPU Backend:

```shell
# The example uses PIL and requests to get the image.
pip install requests pillow
# Run ResNet18 as a standalone script.
python examples/torchscript_resnet18.py

load image from https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /home/mlir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100.0%
PyTorch prediction
[('Labrador retriever', 70.66319274902344), ('golden retriever', 4.956596374511719), ('Chesapeake Bay retriever', 4.195662975311279)]
torch-mlir prediction
[('Labrador retriever', 70.66320037841797), ('golden retriever', 4.956601619720459), ('Chesapeake Bay retriever', 4.195651531219482)]
```

Jupyter notebook:
```shell
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH "$PYTHONPATH"
# Open in jupyter, and then navigate to
# `examples/resnet_inference.ipynb` and use the `torch-mlir` kernel to run.
jupyter notebook
```

### LazyTensorCore

The LazyTensorCore integration is still in progress, and is being built on the
[`torch_mlir_ltc_backend` branch](https://github.com/llvm/torch-mlir/tree/torch_mlir_ltc_backend).

### Eager Mode

Eager mode with TorchMLIR is a very experimental eager mode backend for PyTorch through the torch-mlir framework. 
Effectively, this mode works by compiling operator by operator as the NN is eagerly executed by PyTorch. 
This mode includes a fallback to conventional PyTorch if anything in the torch-mlir compilation process fails (e.g., unsupported operator).
A simple example can be found at [eager_mode.py](examples/eager_mode.py).
A ResNet18 example can be found at [eager_mode_resnet18.py](examples/eager_mode_resnet18.py).

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

## Build Python Packages

We have preliminary support for building Python packages. This can be done
with the following commands:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel
```
