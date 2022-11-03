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

[![Release Build](https://github.com/llvm/torch-mlir/actions/workflows/buildRelease.yml/badge.svg)](https://github.com/llvm/torch-mlir/actions/workflows/buildRelease.yml)

## All the roads from PyTorch to Torch MLIR Dialect

We have few paths to lower down to the Torch MLIR Dialect.

![Simplified Architecture Diagram for README](docs/images/readme_architecture_diagram.png)

 - TorchScript
    This is the most tested path down to Torch MLIR Dialect, and the PyTorch ecosystem is converging on using TorchScript IR as a lingua franca.
 - LazyTensorCore
    Read more details [here](docs/ltc_backend.md).
## Project Communication

- `#torch-mlir` channel on the LLVM [Discord](https://discord.gg/xS7Z362) - this is the most active communication channel
- Github issues [here](https://github.com/llvm/torch-mlir/issues)
- [`torch-mlir` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/torch-mlir/41) of LLVM Discourse
- Weekly meetings on Mondays 9AM PST. See [here](https://discourse.llvm.org/t/community-meeting-developer-hour-refactoring-recurring-meetings/62575) for more information.
- Weekly op office hours on Thursdays 8:30-9:30AM PST. See [here](https://discourse.llvm.org/t/announcing-torch-mlir-office-hours/63973/2) for more information.

## Install torch-mlir snapshot

This installs a pre-built snapshot of torch-mlir for Python 3.7/3.8/3.9/3.10 on Linux and macOS.

```shell
python -m venv mlir_venv
source mlir_venv/bin/activate
# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
pip install --pre torch-mlir torchvision -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# This will install the corresponding torch and torchvision nightlies
```

## Demos

### TorchScript ResNet18 

Standalone script to Convert a PyTorch ResNet18 model to MLIR and run it on the CPU Backend:

```shell
# Get the latest example if you haven't checked out the code
wget https://raw.githubusercontent.com/llvm/torch-mlir/main/examples/torchscript_resnet18.py

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

### Lazy Tensor Core

View examples [here](docs/ltc_examples.md).

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

## Developers
If you would like to develop and build torch-mlir from source please look at [Development Notes](docs/development.md)
