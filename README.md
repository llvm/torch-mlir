# NPComp - MLIR based compiler toolkit for numerical python programs

> This project is participating in the LLVM Incubator process: as such, it is
not part of any official LLVM release.  While incubation status is not
necessarily a reflection of the completeness or stability of the code, it
does indicate that the project is not yet endorsed as a component of LLVM.

The NPComp project aims to provide tooling for compiling numerical python programs of various forms to take advantage of MLIR+LLVM code generation and backend runtime systems.

In addition to providing a bridge to a variety of Python based numerical programming frameworks, NPComp also directly develops components for tracing and compilation of generic Python program fragments.

## Framework integrations

* [PyTorch](frontends/pytorch/README.md) -- Experimental integration for
  extracting programs from PyTorch.

## Python language compiler tookit

At the core of NPComp are a set of dialects and python support code for tracing (define by run) numerical programs and compiling idiomatic subsets of the Python language. As another interpretation of the name, NPComp also seeks to provide compiler-backed support for Numpy APIs.

See the [features doc](docs/features.md) for a semi-curated status of what is implemented in this area.

### Architecture

The compiler is separated into:

* [Frontend importer](python/npcomp/compiler/frontend.py): Translates from
  various AST levels to corresponding MLIR dialects.
* Frontend compiler: MLIR passes and conversions, mostly operating on the
  [basicpy](include/Dialect/Basicpy/IR/BasicpyOps.td) and
  [numpy](include/Dialect/Numpy/IR/NumpyOps.td) dialects.
* Backend compiler and runtime: Some effort has been taken to make this
  pluggable, but right now, only the [IREE Backend](python/npcomp/compiler/backend/iree.py)
  exists. There is in-tree work to also build a minimal reference backend
  directly targeting LLVM.

## Repository Layout

The project is roughly split into the following areas of code:

* [User-facing Python code](python/npcomp)
* C++ [include](include) and [lib](lib) trees, following LLVM/MLIR conventions
* LIT testing trees:
  * [test](test): Lit/FileCheck tests covering core MLIR based infra
  * [test/Python/Compiler](test/Python/Compiler): Lit test suite that drive the compiler
    infra from Python
  * [backend_test](backend_test): Lit test suites conditionally enabled for
    each backend
* [tools](tools): Scripts and binaries (npcomp-opt, npcomp-run-mlir, etc)

## Interactive Use

The cmake configuration populates symlinks in the `build/python` directory
mirroring the source layout. This allows edit-run without rebuilding (unless
if files are added/removed).

Configuring the `PYTHONPATH` as above should be sufficient to run any
interactive tooling (`python3`, Jupyter/Colab, etc).

Note that running the `cmake_configure.sh` script will also output a `.env`
file in the workspace folder with the correct PYTHONPATH set. This allows
tools like VSCode to work by default for debugging.

Notes:

* Python sources are symlinked to the output directory at configure time.
  Adding sources will require a reconfigure. Editing should not.
* It is a very common issue to have both python 2.7 (aka. "python") and python
  3.x (aka. "python3") on a system at a time (and we can only hope that one
  day this ends). Since the native library at development time binds to a
  specific version, if you try to run with a different python, you will get
  an error about the "native" module not being found.

## Compiler development

For bash users, adding the following to your `.bashrc` defines some aliases
that are useful during compiler development, such as shortcuts for builing
and running `npcomp-opt`.

```
source $WHERE_YOU_CHECKED_OUT_NPCOMP/tools/bash_helpers.sh
```

## Build Instructions

### Common prep

```shell
# From checkout directory.
git submodule init
git submodule update

# Use clang and lld to build (optional but recommended).
LLVM_VERSION=10
export CC=clang-$LLVM_VERSION
export CXX=clang++-$LLVM_VERSION
export LDFLAGS=-fuse-ld=$(which ld.lld-$LLVM_VERSION)

# Build and install LLVM/MLIR into the ./install-mlir directory
./build_tools/install_mlir.sh
```

### Vanilla - numpy-only, no pytorch

```shell
# Follow common prep above.
./build_tools/cmake_configure.sh

# Build and run tests
# ./build_tools/test_all.sh runs all of these commands.
cd build
ninja
ninja check-npcomp

# Setup PYTHONPATH for interactive use
export PYTHONPATH="$(realpath python):$(realpath build/python)"
```

### PyTorch 1.3 - ATen pseudo-device type dispatch

The currently functional approach to PyTorch integration uses an ATen pseudo
device for program capture. It is activated by including the PyTorch cmake
path and settind `-DNPCOMP_ENABLE_TORCH_TYPE_DISPATCH=ON`. This approach has a
very fragile dependency on a specific PyTorch revisions in the ~1.3 era and
currently must be built via the docker image in `docker/pytorch-1.3`.

We are migrating to newer approaches that build with more recent PyTorch
versions, but these are not yet functional (see below).

Docker container setup:

```shell
# One of the maintainers does periodically push new images. To use one of these,
# skip the build step and use:
#   BUILD_IMAGE_TAG="stellaraccident/npcomp:build-pytorch-1.3"
# Since we are not planning to support this branch long term, this process is
# entirely ad-hoc at present and geared for project maintainers and build bots
# to be able to make progress.
# See https://hub.docker.com/repository/docker/stellaraccident/npcomp
BUILD_IMAGE_TAG="local/npcomp:build-pytorch-1.3"

# Build the docker image (rebuilds PyTorch, so takes quite some time).
docker build docker/pytorch-1.3 --tag $BUILD_IMAGE_TAG

# Docker workflow (or use your own preferences).
# Create a volume for npcomp build artifacts.
docker volume create npcomp-pytorch-1.3-build

# Run the container, mounting /npcomp to the source directory and the volume
# above to the /build directory. The source directory is mounted read-only to
# avoid the container putting root owned files there.
# Replace `$HOME/src/mlir-npcomp` with an appropriate path to where the project
# is checked out.
docker run \
  --mount type=bind,source=$HOME/src/mlir-npcomp,target=/npcomp,readonly \
  --mount source=npcomp-pytorch-1.3-build,target=/build \
  --rm -it $BUILD_IMAGE_TAG /bin/bash
```

```shell
# From within the docker image.
# Install MLIR and configure project.
cd /npcomp
BUILD_DIR=/build ./build_tools/install_mlir.sh
BUILD_DIR=/build ./build_tools/cmake_configure.sh \
  -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.6/site-packages/torch/share/cmake \
  -DNPCOMP_ENABLE_TORCH_TYPE_DISPATCH=ON

# Build.
cd /build
ninja
ninja check-npcomp
ninja check-frontends-pytorch
```

### PyTorch 1.7+ - Graph API <-> MLIR

TODO
