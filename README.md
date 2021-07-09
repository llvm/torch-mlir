# NPComp - MLIR based compiler toolkit for numerical python programs

> This project is participating in the LLVM Incubator process: as such, it is
not part of any official LLVM release.  While incubation status is not
necessarily a reflection of the completeness or stability of the code, it
does indicate that the project is not yet endorsed as a component of LLVM.

The NPComp project aims to provide tooling for compiling numerical python programs of various forms to take advantage of MLIR+LLVM code generation and backend runtime systems.

In addition to providing a bridge to a variety of Python based numerical programming frameworks, NPComp also directly develops components for tracing and compilation of generic Python program fragments.

## Quick Build and Install with PyTorch

Instructions below go in to detail about how to get a functioning development
setup. But to just build and install into a local python instance, the
instructions are simple. This support is new and we still need to get
dependencies pinned and make these packages distributable. This should work
locally, though:

```
python -m pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

./build_tools/build_python_wheels.sh
```

## Project Communication

- `#mlir-npcomp` channel on the LLVM [Discord](https://discord.gg/xS7Z362)
- issues/PR's on this github repo
- [`mlir-npcomp` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/mlir-npcomp/41) of LLVM Discourse

## Framework integrations

* [PyTorch](frontends/pytorch/README.md) -- Experimental integration for
  extracting programs from PyTorch.

## Python language compiler tookit

At the core of NPComp are a set of dialects and python support code for tracing (define by run) numerical programs and compiling idiomatic subsets of the Python language. As another interpretation of the name, NPComp also seeks to provide compiler-backed support for Numpy APIs.

See the [features doc](docs/features.md) for a semi-curated status of what is implemented in this area.

### Architecture

The compiler is separated into:

* [Frontend importer](python/npcomp/compiler/numpy/frontend.py): Translates from
  various AST levels to corresponding MLIR dialects.
* Frontend compiler: MLIR passes and conversions, mostly operating on the
  [basicpy](include/npcomp/Dialect/Basicpy/IR/BasicpyOps.td) and
  [numpy](include/npcomp/Dialect/Numpy/IR/NumpyOps.td) dialects.
* Backend compiler and runtime: Some effort has been taken to make this
  pluggable, but right now, only the [IREE Backend](python/npcomp/compiler/generic/backend/iree.py)
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

For using with any interactive tooling (`python3`, Jupyter/Colab, etc) it should be
sufficient to add `python_packages/npcomp_torch/` and `python_packages/npcomp_core/`
directories under `build` directory to `PYTHONPATH`

```shell
export PYTHONPATH=$(cd build && pwd)/python_packages/npcomp_torch:$(cd build && pwd)/python_packages/npcomp_core
```

Note that running the `build_tools/write_env_file.sh` script will output a `.env`
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

# run write_env_file.sh to emit a .env file with needed
# PYTHONPATH setup.
./build_tools/write_env_file.sh
```

### Vanilla - numpy-only, no pytorch

```shell
# Configure npcomp.
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release .

# Build and run tests
# ./build_tools/test_all.sh runs all of these commands.
cd build
ninja
ninja check-npcomp
```

### With PyTorch integration

```shell
# Install PyTorch. We currently track and require the nighly build.
# If a usable PyTorch package is installed, the default cmake settings will
# enable the PyTorch frontend.
pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

cmake -DNPCOMP_ENABLE_PYTORCH=ON ...
ninja check-frontends-pytorch  # If building with PyTorch
```

### PyTorch Frontend (via docker container)

Create docker image (or follow your own preferences):

* Mount the (host) source directory to `/src/mlir-npcomp` (in the container).
* Mount the `/build` directory (in the container) appropriately for your case.

```shell
docker build docker/pytorch-nightly --tag local/npcomp:build-pytorch-nightly
docker volume create npcomp-build
```

Shell into docker image:

```shell
docker run \
  --mount type=bind,source=$HOME/src/mlir-npcomp,target=/src/mlir-npcomp \
  --mount source=npcomp-build,target=/build \
  --rm -it local/npcomp:build-pytorch-nightly /bin/bash
```

Build/test npcomp (from within docker image):

```shell
# From within the docker image.
cd /src/mlir-npcomp
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DNPCOMP_ENABLE_PYTORCH=ON .
cmake --build /build/npcomp --target check-npcomp check-frontends-pytorch
```

### IREE Backend (from IREE packages)

```shell
# We currently track and require the latest snapshot.
pip3 install iree-compiler-snapshot iree-runtime-snapshot -f https://github.com/google/iree/releases

# Run TorchScript E2E tests targeting IREE.
# Make sure to run "PyTorch Frontend" setup instructions first.
tools/torchscript_e2e_test.sh --config=iree
```

### IREE Backend (from local IREE build)

This configuration is useful for iterating locally, as you can
poke/debug/rebuild things in IREE.

```shell
# Locally build IREE.
# See https://google.github.io/iree/building-from-source/getting-started/
# Make sure IREE is configured with `-DIREE_BUILD_PYTHON_BINDINGS=ON`.

# Add IREE's Python bindings to PYTHONPATH.
echo 'PYTHONPATH="${PYTHONPATH}:/path/to/iree-build/bindings/python"' >> .env

# Run TorchScript E2E tests targeting IREE.
# Make sure to run "PyTorch Frontend" setup instructions first.
tools/torchscript_e2e_test.sh --config=iree
```

### Additional end-to-end tests with heavy dependencies (heavydep tests)

Some end-to-end tests require additional dependencies which don't make sense to
include as part of the default npcomp setup. Additionally, these dependencies
often don't work with the same HEAD PyTorch version that npcomp builds against
at the C++ level.

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
tools/torchscript_e2e_test.sh --serialized-test-dir=/t/heavydep_serialized_tests
```

Note that the heavy dep tests are generally quite challenging, and we don't have
any that work yet. The tests use the same (pure-Python) test framework as the
normal torchscript_e2e_test.sh, but the tests are added in
`build_tools/torchscript_e2e_heavydep_tests` instead of
`frontends/pytorch/e2e_testing/torchscript`.

We rely critically on serialized TorchScript compatibility across PyTorch
versions to transport the tests + pure-Python compatibility of the `torch`
API, which has worked well so far.

### VSCode with a Docker Dev Image

#### Start a docker dev container based on our image

Assumes that mlir-npcomp is checked out locally under `~/src/mlir-npcomp`.
See `docker_shell_funcs.sh` for commands to modify if different.

```shell
# Build/start the container.
# Follow instructions here to allow running `docker` without `sudo`:
# https://docs.docker.com/engine/install/linux-postinstall/
source ./build_tools/docker_shell_funcs.sh
npcomp_docker_build  # Only needed first time/on updates to docker files.
npcomp_docker_start
```

```shell
# Get an interactive shell to the container and initial build.
npcomp_docker_login
```

```shell
# Stop the container (when done).
npcomp_docker_stop
```

### Configure VSCode:

First, install the [VSCode Docker
extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) and [VSCode Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
Follow instructions here to allow running `docker` without `sudo`,
otherwise VSCode won't be able to use docker
https://docs.docker.com/engine/install/linux-postinstall/
(Note that VSCode has some daemons that you will need to kill/restart for
the instructions there to take effect; consider just rebooting your
machine)

Attach to your running container by opening the Docker extension tab (left panel), right clicking on the container name, and selecting "Attach Visual Studio code". The container name if you are using docker_shell_funcs.sh is `npcomp`.

Install extensions in container:
  * CMake Tools
  * C/C++
  * C++ Intellisense

#### Add workspace folders:

* `mlir-npcomp` source folder
* `external/llvm-project` source folder

#### Configure general settings:

`Ctrl-Shift-P` > `Preferences: Open Settings (UI)`

* For `mlir-npcomp` folder:
  * `Cmake: Build directory`: `/build/npcomp`
  * Uncheck `Cmake: Configure On Edit` and `Cmake: Configure on Open`
* For `llvm-project` folder:
  * `Cmake: Build directory`: `/build/llvm-build`
  * Uncheck `Cmake: Configure On Edit` and `Cmake: Configure on Open`

#### Configure Intellisense:

`Ctrl-Shift-P` > `C/C++: Edit Configurations (UI)`

* Open C/C++ config (for each project folder):
  * Under Advanced, Compile Commands:
    * set `/build/npcomp/compile_commands.json` for mlir-npcomp
  	* set `/build/llvm-build/compile_commands.json` for llvm-project
* Open a C++ file, give it a few seconds and see if you get code completion
  (press CTRL-Space).

Make sure to save your workspace (prefer a local folder with the "Use Local" button)!
