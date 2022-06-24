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

Torch-MLIR can also be built using Bazel (apart from the official CMake build) for users that depend on Bazel in their workflows. To build `torch-mlir-opt` using Bazel, follow these steps:

1. Install [Bazel](https://docs.bazel.build/versions/main/install.html) if you don't already have it
2. Install a relatively new release of [Clang](https://releases.llvm.org/download.html)
3. Build:
```shell
cd utils/bazel
bazel build @torch-mlir//...
```
4. Find the built binary at `bazel-bin/external/torch-mlir/torch-mlir-opt`.


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
./tools/torchscript_e2e_test.sh
# Run tests that match the regex `Conv2d`, with verbose errors.
./tools/torchscript_e2e_test.sh --filter Conv2d --verbose
# Run tests on the TOSA backend.
./tools/torchscript_e2e_test.sh --config tosa
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
# Updating the LLVM submodule

Torch-MLIR maintains `llvm-project` (which contains, among other things,
upstream MLIR) as a submodule in `externals/llvm-project`. We aim to update this
at least weekly to new LLVM revisions to bring in the latest features and spread
out over time the effort of updating our code for MLIR API breakages.

Updating the LLVM submodule is done by:

1. In the `externals/llvm-project` directory, run `git pull` to update to the
   upstream revision of interest (such as a particular upstream change that is
   needed for your Torch-MLIR PR).
2. Rebuild and test Torch-MLIR (see above), fixing any issues that arise. This
   might involve fixing various API breakages introduced upstream (they are
   likely unrelated to what you are working on). If these fixes are too complex,
   please file a work-in-progress PR explaining the issues you are running into
   asking for help so that someone from the community can help.
3. Run `build_tools/update_shape_lib.sh` to update the shape library -- this is
   sometimes needed because upstream changes can affect canonicalization and
   other minor details of the IR in the shape library. See [docs/shape_lib.md](docs/shape_lib.md) for more details on the shape library.


Here are some examples of PR's updating the LLVM submodule:

- https://github.com/llvm/torch-mlir/pull/958
- https://github.com/llvm/torch-mlir/pull/856
