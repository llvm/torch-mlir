# Checkout and build from source

## Check out the code

```shell
git clone https://github.com/llvm/torch-mlir
cd torch-mlir
git submodule update --init
```

## Setup your Python VirtualEnvironment and Dependencies

Also, ensure that you have the appropriate `python-dev` package installed
to access the Python development libraries / headers.

```shell
python -m venv mlir_venv
source mlir_venv/bin/activate
# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
# Install latest PyTorch nightlies and build requirements.
python -m pip install -r requirements.txt
```

## Docker Builds

We have preliminary support for building with Docker images. This is a new
flow and we would like your feedback on how it works for you and please
feel free to file any feedback or issues.

Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/). You don't need Docker Desktop.

You have three types of builds selectable with the Environment Variable `TM_PACKAGES`:`torch-mlir` the
Release build, `out-of-tree` where torch-mlir is build with a pre-built MLIR and `in-tree` where torch-mlir
is built as part of the LLVM project along with MLIR.

We mount a ccache and pip cache inside the docker container to speed up iterative builds. Iterative
builds should be as fast as running without docker.

### In-Tree builds

Build MLIR and Torch-MLIR together as part of the LLVM repo.

```shell
TM_PACKAGES="in-tree" ./build_tools/python_deploy/build_linux_packages.sh
```

### Out-of-Tree builds

Build LLVM/MLIR first and then build Torch-MLIR referencing that build
```shell
TM_PACKAGES="out-of-tree" ./build_tools/python_deploy/build_linux_packages.sh
```

### Release builds

Build in a manylinux Docker image so we can upload artifacts to PyPI.

```shell
TM_PACKAGES="torch-mlir" ./build_tools/python_deploy/build_linux_packages.sh
```

### Mimicing CI+Release builds

If you wanted to build all the CIs and the Release builds (just with Python 3.10 since most other Python builds are redundant)

```shell
TM_PACKAGES="torch-mlir out-of-tree in-tree" TM_PYTHON_VERSIONS="cp310-cp310" ./build_tools/python_deploy/build_linux_packages.sh
```

Note: The Release docker still runs as root so it may generate some files owned by root:root. We hope to move it to run as a user in the future.

### Cleaning up

Docker builds tend to leave a wide variety of files around. Luckily most are owned by the user but there are still some that need to be removed
as superuser.

```shell
rm -rf build build_oot llvm-build docker_venv externals/pytorch/build .ccache
```

### Other configurable environmental variables

The following additional environmental variables can be used to customie your docker build:

* Custom Release Docker image:
  Defaults to `stellaraccident/manylinux2014_x86_64-bazel-5.1.0:latest`
```shell
  TM_RELEASE_DOCKER_IMAGE="stellaraccident/manylinux2014_x86_64-bazel-5.1.0:latest"
```
* Custom CI Docker image:
  Defaults to `powderluv/torch-mlir-ci:latest`. This assumes an Ubuntu LTS like image. You can build your own with `./build_tools/docker/Dockerfile`
```shell
  TM_CI_DOCKER_IMAGE="powderluv/torch-mlir-ci:latest"
```

* Custom Python Versions for Release builds:
  Version of Python to use in Release builds. Ignored in CIs. Defaults to `cp38-cp38 cp39-cp39 cp310-cp310`
```shell
  TM_PYTHON_VERSIONS="cp38-cp38 cp39-cp39 cp310-cp310"
```

* Location to store Release build wheels
```shell
  TM_OUTPUT_DIR="./build_tools/python_deploy/wheelhouse"
```

* What "packages" to build:
  Defaults to torch-mlir. Options are `torch-mlir out-of-tree in-tree`
```shell
  TM_PACKAGES="torch-mlir out-of-tree in-tree"
```
* Use pre-built Pytorch:
  Defaults to using pre-built Pytorch. Setting it to `OFF` builds from source
```shell
  TM_USE_PYTORCH_BINARY="OFF"
```
* Skip running tests
  Skip running tests if you want quick build only iteration. Default set to `OFF`
```shell
  TM_SKIP_TESTS="OFF"
```


## Build Python Packages

We have preliminary support for building Python packages. This can be done
with the following commands:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel
```

## CMake Build

Two setups are possible to build: in-tree and out-of-tree. The in-tree setup is the most straightforward, as it will build LLVM dependencies as well.

### Building torch-mlir in-tree

The following command generates configuration files to build the project *in-tree*, that is, using llvm/llvm-project as the main build. This will build LLVM as well as torch-mlir and its subprojects.

```shell
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
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
```
The following additional quality of life flags can be used to reduce build time:
* Enabling ccache:
```shell
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```
* Enabling LLD (links in seconds compared to minutes)
```shell
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
# Use --ld-path= instead of -fuse-ld=lld for clang > 13
```
* Enabling libtorch binary cache
By default we download the latest version of libtorch everytime you build so we are always on the latest version. Set `-DLIBTORCH_CACHE=ON` to
not download the latest version everytime. If libtorch gets out of date and you test against a newer PyTorch you may notice failures.
```shell
  -DLIBTORCH_CACHE=ON
```
* Enabling building libtorch as part of your build
By default we download the latest version of libtorch. We have an experimental path to build libtorch (and PyTorch wheels) from source.
```shell
  -DLIBTORCH_SRC_BUILD=ON  # Build Libtorch from source
  -DLIBTORCH_VARIANT=shared # Set the variant of libtorch to build / link against. (`shared`|`static` and optionally `cxxabi11`)
```

### Building against a pre-built LLVM

If you have built llvm-project separately in the directory `$LLVM_INSTALL_DIR`, you can also build the project *out-of-tree* using the following command as template:
```shell
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  .
```
The same QoL CMake flags can be used to enable ccache and lld. Be sure to have built LLVM with `-DLLVM_ENABLE_PROJECTS=mlir`.

Be aware that the installed version of LLVM needs in general to match the committed version in `externals/llvm-project`. Using a different version may or may not work.


### Build commands

After either cmake run (in-tree/out-of-tree), use one of the following commands to build the project:
```shell
# Build just torch-mlir (not all of LLVM)
cmake --build build --target tools/torch-mlir/all

# Run unit tests.
cmake --build build --target check-torch-mlir

# Run Python regression tests.
cmake --build build --target check-torch-mlir-python

# Build everything (including LLVM if in-tree)
cmake --build build
```

## Setup Python Environment to export the built Python packages
```shell
export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples
```

## Testing MLIR output in various dialects

To test the compiler's output to the different MLIR dialects, you can use the example `examples/torchscript_resnet18_all_output_types.py`.

Make sure you have activated the virtualenv and set the `PYTHONPATH` above:
```shell
source mlir_venv/bin/activate
export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples
python examples/torchscript_resnet18_all_output_types.py
```

This will display the Resnet18 network example in three dialects: TORCH, LINALG on TENSORS and TOSA.

The main functionality is on `torch_mlir.compile()`'s `output_type`.

Ex:
```python
module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="torch")
```

Currently, `output_type` can be: `TORCH`, `TOSA`, `LINALG_ON_TENSORS`, `RAW` and `MHLO`.

## Jupyter

Jupyter notebook:
```shell
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH "$PYTHONPATH"
# Open in jupyter, and then navigate to
# `examples/resnet_inference.ipynb` and use the `torch-mlir` kernel to run.
jupyter notebook
```

[Example IR](https://gist.github.com/silvasean/e74780f8a8a449339aac05c51e8b0caa) for a simple 1 layer MLP to show the compilation steps from TorchScript.


## Interactive Use

The `build_tools/write_env_file.sh` script will output a `.env`
file in the workspace folder with the correct PYTHONPATH set. This allows
tools like VSCode to work by default for debugging. This file can also be
manually `source`'d in a shell.


## Bazel Build

> **NOTE** Our Bazel build follows LLVM's Bazel build policy: only the
> subcommunity interested in Bazel is responsible for fixing it. Average
> Torch-MLIR developers should not be notified of any Bazel build issues and are
> not responsible for fixing any breakages (though any help is, of course,
> welcome). For more info, see LLVM's
> [Peripheral Support Tier](https://llvm.org/docs/SupportPolicy.html#peripheral-tier)
> definition.

Torch-MLIR can also be built using Bazel (apart from the official CMake build) for users that depend on Bazel in their workflows. To build `torch-mlir-opt` using Bazel, follow these steps:

1. Launch an interactive docker container with the required deps installed:
```shell
./utils/bazel/docker/run_docker.sh
```
2. Build torch-mlir using bazel (from container):
```shell
./utils/bazel/docker/run_bazel_build.sh
```
3. Find the built binary at `utils/bazel/bazel-bin/external/torch-mlir/torch-mlir-opt`.


# Testing

Torch-MLIR has two types of tests:

1. End-to-end execution tests. These compile and run a program and check the
   result against the expected output from execution on native Torch. These use
   a homegrown testing framework (see
   `python/torch_mlir_e2e_test/torchscript/framework.py`) and the test suite
   lives at `python/torch_mlir_e2e_test/test_suite/__init__.py`.

2. Compiler and Python API unit tests. These use LLVM's `lit` testing framework.
   For example, these might involve using `torch-mlir-opt` to run a pass and
   check the output with `FileCheck`.


## Running execution (end-to-end) tests:

```shell
# Run all tests on the reference backend
./tools/e2e_test.sh
# Run tests that match the regex `Conv2d`, with verbose errors.
./tools/e2e_test.sh --filter Conv2d --verbose
# Run tests on the TOSA backend.
./tools/e2e_test.sh --config tosa
```

## Running unit tests.

To run all of the unit tests, run:

```
ninja check-torch-mlir-all
```

This can be broken down into

```
ninja check-torch-mlir check-torch-mlir-dialects check-torch-mlir-python
```

To run more fine-grained tests, you can do, for `check-torch-mlir`:

```
cd $TORCH_MLIR_BUILD_DIR/tools/torch-mlir/test
$TORCH_MLIR_BUILD_DIR/bin/llvm-lit $TORCH_MLIR_SRC_ROOT/test -v --filter=canonicalize
```

See [the `lit` documentation](https://llvm.org/docs/CommandGuide/lit.html) for details on the available lit args.

For example, if you wanted to test just `test/Dialect/Torch/canonicalize.mlir`,
then you might do

```
cd $TORCH_MLIR_BUILD_DIR/tools/torch-mlir/test
$TORCH_MLIR_BUILD_DIR/bin/llvm-lit $TORCH_MLIR_SRC_ROOT/test -v --filter=canonicalize.mlir
```

Most of the unit tests use the [`FileCheck` tool](https://llvm.org/docs/CommandGuide/FileCheck.html) to verify expected outputs.

# PyTorch source builds and custom PyTorch versions

Torch-MLIR by default builds with the latest nightly PyTorch version. This can be toggled to build from latest PyTorch source with 
```
-DTORCH_MLIR_USE_INSTALLED_PYTORCH=OFF
-DTORCH_MLIR_SRC_PYTORCH_REPO=vivekkhandelwal1/pytorch # Optional. Github path. Defaults to pytorch/pytorch
-DTORCH_MLIR_SRC_PYTORCH_BRANCH=master # Optional. Defaults to PyTorch's main branch
```

# Updating the LLVM and MLIR-HLO submodules

Torch-MLIR depends on `llvm-project` (which contains, among other things,
upstream MLIR) and `mlir-hlo`, both of which are submodules in the `externals/`
directory. We aim to update these at least weekly to bring in the latest
features and spread out over time the effort of updating our code for MLIR API
breakages.

## Which LLVM commit should I pick?

Since downstream projects may want to build Torch-MLIR (and thus LLVM and
MLIR-HLO) in various configurations (Release versus Debug builds; on Linux,
Windows, or macOS; possibly with Clang, LLD, and LLDB enabled), it is crucial to
pick LLVM commits that pass tests for all combinations of these configurations.

So every week, we track the so-called _green_ commit (i.e. the LLVM commit which
works with all of the above configurations) in Issue
https://github.com/llvm/torch-mlir/issues/1178.  In addition to increasing our
confidence that the resulting update will not break downstream projects, basing
our submodule updates on these green commits also helps us stay in sync with
LLVM updates in other projects like ONNX-MLIR and MLIR-HLO.

## What is the update process?

1. **Lookup green commit hashes**: From the Github issue
   https://github.com/llvm/torch-mlir/issues/1178, find the LLVM and MLIR-HLO
   green commits for the week when Torch-MLIR is being updated.
2. **Update the `llvm-project` submodule**: In the `externals/llvm-project`
   directory, run `git fetch` followed by `git checkout <llvm-commit-hash>`
   (where `<llvm-commit-hash>` is the green commit hash for the LLVM project
   from Step 1).
3. **Update the `mlir-hlo` submodule**: In the `externals/mlir-hlo` directory,
   run `git fetch` followed by `git checkout <mlir-hlo-commit-hash>` (where
   `<mlir-hlo-commit-hash>` is the green commit hash for the MLIR-HLO project
   from Step 1).
4. **Rebuild and test Torch-MLIR**: See the section "CMake Build" above for
   instructions, fixing any issues that arise. This might involve fixing various
   API breakages introduced upstream (they are likely unrelated to what you are
   working on).  If these fixes are too complex, please file a work-in-progress
   PR explaining the issues you are running into asking for help so that someone
   from the community can help.
5. **Update Shape Library**: Run `build_tools/update_shape_lib.sh`. This is
   sometimes needed because upstream changes can affect canonicalization and
   other minor details of the IR in the shape library. See
   [docs/shape_lib.md](docs/shape_lib.md) for more details on the shape library.


Here are some examples of PRs updating the LLVM and MLIR-HLO submodules:

- https://github.com/llvm/torch-mlir/pull/1180
- https://github.com/llvm/torch-mlir/pull/1229

# Enabling Address Sanitizer (ASan)

To enable ASAN, pass `-DLLVM_USE_SANITIZER=Address` to CMake. This should "just
work" with all C++ tools like `torch-mlir-opt`. When running a Python script
such as through `./tools/e2e_test.sh`, you will need to do:

```
LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)" ./tools/e2e_test.sh -s
# See instructions here for how to get the libasan path for GCC:
# https://stackoverflow.com/questions/48833176/get-location-of-libasan-from-gcc-clang
```

TODO: Add ASan docs for LTC.

# Other docs

- GitHub wiki: https://github.com/llvm/torch-mlir/wiki
- Of particular interest in the [How to add end-to-end support for new Torch ops](https://github.com/llvm/torch-mlir/wiki/Torch-ops-E2E-implementation) doc.
