# Development Guide

## Setting Up Environment

### Clone the Repository

1. ```shell
   git clone https://github.com/llvm/torch-mlir && cd torch-mlir
   ```

1. ```shell
   git submodule update --init --progress
   ```

   - Optionally, use `--depth=1` to make a shallow clone of the submodules.
   - While this is running, you can already setup the Python venv and dependencies in the next step.

### Set up the Python environment

1. Install Python development libraries and headers

    - For Ubuntu or Debian, run:

      ```shell
      sudo apt install python3-dev
      ```

    - For other operating systems, [download Python](https://www.python.org/downloads)

1. Create and Activate Python VirtualEnvironment + MLIR variant

    ```shell
    python3 -m venv mlir_venv
    source mlir_venv/bin/activate
    ```

1. Get the latest version of pip

    ```shell
    python -m pip install --upgrade pip
    ```

    NOTE: Some older pip installs may not be able to handle the recent PyTorch deps
1. Install the latest requirements.

    ```shell
    python -m pip install -r requirements.txt -r torchvision-requirements.txt
    ```

### Set up pre-commit hooks

We recommend linting and formatting your commits _before_ the CI has a chance to complain about it.

1. Install [pre-commit](https://pre-commit.com/)

    ```shell
    pip install pre-commit
    ```

    - This is the same package used by the CI.
1. Either:
    - Run the hooks manually.

      ```shell
      pre-commit run
      ```

      OR
    - Install them so they run automatically.

      ```shell
      pre-commit install
      ```

## Building

### With CMake

#### (Optional) Enable Quicker Builds

For workflows that demand frequent rebuilds, the following steps will allow you to specify the relevant options during configuration.

##### On Ubuntu

Install [Clang](https://clang.llvm.org/index.html), [ccache](https://ccache.dev/), and [LLD](https://lld.llvm.org/)

```shell
sudo apt install clang ccache lld
```

##### On Windows

  1. Set up Developer PowerShell [for Visual Studio](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022#start-in-visual-studio)
  1. Ensure that the compiler and linker binaries are in the `PATH` variable.

#### Configure for Building

1. **If you haven't already**, [activate the Python environment](#set-up-the-python-environment)
1. Choose command relevant to LLVM setup:
    1. **If you want the more straightforward option**, run the "in-tree" setup:

        ```shell
        cmake -GNinja -Bbuild \
          `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DPython3_FIND_VIRTUALENV=ONLY \
          -DPython_FIND_VIRTUALENV=ONLY \
          -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
          -DLLVM_TARGETS_TO_BUILD=host \
          `# For building LLVM "in-tree"` \
          externals/llvm-project/llvm \
          -DLLVM_ENABLE_PROJECTS=mlir \
          -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
          -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"
        ```

        - NOTE: uses external/llvm-project/llvm as the main build, so LLVM will be built in addition to torch-mlir and its sub-projects.
    1. **If you want to use a separate build of LLVM from another directory**, run the "out-of-tree" setup:

        ```shell
        cmake -GNinja -Bbuild \
          `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DPython3_FIND_VIRTUALENV=ONLY \
          -DPython_FIND_VIRTUALENV=ONLY \
          -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
          -DLLVM_TARGETS_TO_BUILD=host \
          `# For building LLVM "out-of-tree"` \
          -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
          -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/"
        ```

        - Be sure to have built LLVM with `-DLLVM_ENABLE_PROJECTS=mlir`.
        - Be aware that the installed version of LLVM needs in general to match the committed version in `externals/llvm-project`. Using a different version may or may not work.

    - [About MLIR debugging](https://mlir.llvm.org/getting_started/Debugging/)
1. **If you anticipate needing to frequently rebuild LLVM**, leverage quicker builds by appending:

    ```shell
      \
      `# use clang`\
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      `# use ccache to cache build results` \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      `# use LLD to link in seconds, rather than minutes` \
      -DCMAKE_LINKER_TYPE=lld
    ```

    - This requires [the enablement mentioned earlier](#optional-enable-quicker-builds).
    - If these flags cause issues, just skip them for now.
1. **If you're developing changes**, enable local end-to-end tests by appending:

    ```shell
      \
      -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
      -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
    ```

    - NOTE: The JIT IR importer depends on the native PyTorch extension features and defaults to `ON` if not changed.

#### Initiate Build

1. [Configure the build](#configure-for-building) if you haven't already done so.
1. **If you want to...**
    - **...build _everything_** (including LLVM if configured as "in-tree"), run:

      ```shell
      cmake --build build
      ```

    - **...build _just_ torch-mlir** (not all of LLVM), run:

      ```shell
      cmake --build build --target tools/torch-mlir/all
      ```

    - **...run unit tests**, run:

      ```shell
      cmake --build build --target check-torch-mlir
      ```

    - **...run Python regression tests**, run:

      ```shell
      cmake --build build --target check-torch-mlir-python
      ```

TIP: add multiple target options to stack build phases

### Setup Python Environment to export the built Python packages

When CMake is configured with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`, the python packages will typically be located in either:

1. `./build/tools/torch-mlir/python_packages/`  if doing an in-tree build.
2. `./build/python_packages/` if doing an out-of-tree build.

For the following sections, let `python_pkg_dir` represent whichever of the above is relevant for your build setup. On Linux and macOS, you can run `./build_tools/write_env_file.sh` to generate a file `./.env` in your root source directory with the correct `PYTHONPATH`.

#### Linux and macOS

To get the base `PYTHONPATH`, run:

```shell
./build_tools/write_env_file.sh
source ./.env && export PYTHONPATH
```

To run fx_importer tests, you can append the following:

```
export PYTHONPATH="${PYTHONPATH}":/test/python/fx_importer"
```

#### Windows PowerShell

To get the base `PYTHONPATH`, identify your `python_pkg_dir` from above and set this variable in your environment:

```shell
$env:PYTHONPATH = "<python_pkg_dir>/torch-mlir"
```

To run fx_importer tests, you can append the following:

```shell
$env:PYTHONPATH += ";$PWD/test/python/fx_importer"
```

### Testing MLIR output in various dialects

To test the MLIR output to torch dialect, you can use `test/python/fx_importer/basic_test.py`.

Make sure you have activated the virtualenv and set the `PYTHONPATH` above
(if running on Windows, modify the environment variable as shown above).

This will display the basic example in TORCH dialect.

To test the compiler's output to the different MLIR dialects, you can also use the deprecated path
using torchscript with the example `projects/pt1/examples/torchscript_resnet18_all_output_types.py`.
This path doesn't give access to the current generation work that is being driven via the fx_importer
and may lead to errors.

The base `PYTHONPATH` should be set as above, then the example can be run with the following command (similar on Windows):

```shell
export PYTHONPATH="${PYTHONPATH}:$PWD/projects/pt1/examples"
python projects/pt1/examples/torchscript_resnet18_all_output_types.py
```

This will display the Resnet18 network example in three dialects: TORCH, LINALG on TENSORS and TOSA.

The main functionality is on `torch_mlir.torchscript.compile()`'s `output_type`.

Ex:

```python
module = torch_mlir.torchscript.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="torch")
```

`output_type` can be: `TORCH`, `TOSA`, `LINALG_ON_TENSORS`, `RAW` and `STABLEHLO`.

### Jupyter

Jupyter notebook:

```shell
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH "$PYTHONPATH"
# Open in jupyter, and then navigate to
# `examples/resnet_inference.ipynb` and use the `torch-mlir` kernel to run.
jupyter notebook
```

[Example IR](https://gist.github.com/silvasean/e74780f8a8a449339aac05c51e8b0caa) for a simple 1 layer MLP to show the compilation steps from TorchScript.


### Interactive Use

The `build_tools/write_env_file.sh` script will output a `.env`
file in the workspace folder with the correct PYTHONPATH set. This allows
tools like VSCode to work by default for debugging. This file can also be
manually `source`'d in a shell.


### Bazel Build

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

2. Build torch-mlir:

    ```shell
    bazel build @torch-mlir//:torch-mlir-opt
    ```

    The built binary should be at `bazel-bin/external/torch-mlir/torch-mlir-opt`.

3. Test torch-mlir (lit test only):

    ```shell
    bazel test @torch-mlir//test/...
    ```

We welcome patches to torch-mlir's Bazel build. If you do contribute,
please complete your PR with an invocation of buildifier to ensure
the BUILD files are formatted consistently:

```shell
bazel run //utils/bazel:buildifier
```

### Docker Builds

We have preliminary support for building with Docker images. Currently this
is not very convenient for day-to-day interactive development and
debugging flows but is very useful for reproducing failures
from the CI. This is a new flow and we would like your feedback on how
it works for you and please feel free to file any feedback or issues.

Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/). You don't need Docker Desktop.

You have three types of builds selectable with the Environment Variable `TM_PACKAGES`:`torch-mlir` the
Release build, `out-of-tree` where torch-mlir is build with a pre-built MLIR and `in-tree` where torch-mlir
is built as part of the LLVM project along with MLIR.

We mount a ccache and pip cache inside the docker container to speed up iterative builds. Iterative
builds should be as fast as running without docker.

#### In-Tree builds

Build MLIR and Torch-MLIR together as part of the LLVM repo.

```shell
TM_PACKAGES="in-tree" ./build_tools/python_deploy/build_linux_packages.sh
```

#### Out-of-Tree builds

Build LLVM/MLIR first and then build Torch-MLIR referencing that build

```shell
TM_PACKAGES="out-of-tree" ./build_tools/python_deploy/build_linux_packages.sh
```

#### Release builds

Build in a manylinux Docker image so we can upload artifacts to PyPI.

```shell
TM_PACKAGES="torch-mlir" ./build_tools/python_deploy/build_linux_packages.sh
```

#### Mimicing CI+Release builds

If you wanted to build all the CIs locally

```shell
TM_PACKAGES="out-of-tree in-tree" ./build_tools/python_deploy/build_linux_packages.sh
```

If you wanted to build all the CIs and the Release builds (just with Python 3.10 since most other Python builds are redundant)

```shell
TM_PACKAGES="torch-mlir out-of-tree in-tree" TM_PYTHON_VERSIONS="cp310-cp310" ./build_tools/python_deploy/build_linux_packages.sh
```

Note: The Release docker still runs as root so it may generate some files owned by root:root. We hope to move it to run as a user in the future.

#### Cleaning up

Docker builds tend to leave a wide variety of files around. Luckily most are owned by the user but there are still some that need to be removed
as superuser.

```shell
rm -rf build build_oot llvm-build docker_venv externals/pytorch/build .ccache
```

### Building your own Docker image

If you would like to build your own docker image (usually not necessary). You can run:

```shell
cd ./build_tools/docker
docker build -t your-name/torch-mlir-ci --no-cache .
```

#### Other configurable environmental variables

The following additional environmental variables can be used to customize your docker build:

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
  Version of Python to use in Release builds. Ignored in CIs. Defaults to `cp39-cp39 cp310-cp310 cp312-cp312`

    ```shell
      TM_PYTHON_VERSIONS="cp39-cp39 cp310-cp310 cp312-cp312"
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


### Build Python Packages

We have preliminary support for building Python packages. This can be done
with the following commands:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel
```

To package a completed CMake build directory,
you can use the `TORCH_MLIR_CMAKE_BUILD_DIR` and `TORCH_MLIR_CMAKE_ALREADY_BUILT` environment variables:

```shell
TORCH_MLIR_CMAKE_BUILD_DIR=build/ TORCH_MLIR_CMAKE_ALREADY_BUILT=1 python setup.py bdist_wheel
```

Note: The setup.py script is only used for building the Python packages,
not support commands like `setup.py develop` to build the development environment.

## Testing

Torch-MLIR has two types of tests:

1. End-to-end execution tests. These compile and run a program and check the
   result against the expected output from execution on native Torch. These use
   a homegrown testing framework (see
   `projects/pt1/python/torch_mlir_e2e_test/framework.py`) and the test suite
   lives at `projects/pt1/python/torch_mlir_e2e_test/test_suite/__init__.py`.
   The tests require to build with `TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS` (and
   the dependent option `TORCH_MLIR_ENABLE_JIT_IR_IMPORTER`) set to `ON`.

2. Compiler and Python API unit tests. These use LLVM's `lit` testing framework.
   For example, these might involve using `torch-mlir-opt` to run a pass and
   check the output with `FileCheck`. These tests usually live in the `test/`
   directory with a parallel file naming scheme to the `lib/*` structure.
   More details about this kind of test is available in the upstream
   [LLVM Testing Guide](https://llvm.org/docs/TestingGuide.html#regression-test-structure).


### Running execution (end-to-end) tests:

> **Note**
> An `.env` file must be generated via `build_tools/write_env_file.sh` before these commands can be run.


The following assumes you are in the `projects/pt1`  directory:

```shell
# Run all tests on the reference backend
./tools/e2e_test.sh
# Run tests that match the regex `Conv2d`, with verbose errors.
./tools/e2e_test.sh --filter Conv2d --verbose
# Run tests on the TOSA backend via fx_importer path
./tools/e2e_test.sh --config fx_importer_tosa
```

Alternatively, you can run the tests via Python directly:

```shell
cd projects/pt1
python -m e2e_testing.main -f 'AtenEmbeddingBag'
```

The default mode of running tests uses the multi-processing framework and is
not tolerant of certain types of errors. If encountering native crashes/hangs,
enable debug variables to run sequentially/in-process with more verbosity:

```
export TORCH_MLIR_TEST_CONCURRENCY=1
export TORCH_MLIR_TEST_VERBOSE=1
```

In this way, you can run under `gdb`, etc and get useful results. Having env
vars like this makes it easy to set in GH action files, etc. Note that the
verbose flags are very verbose. Basic sequential progress reports will be
printed regardless when not running in parallel.

### Running unit tests.

To run all of the unit tests, run:

```
ninja check-torch-mlir-all
```

This can be broken down into

```
ninja check-torch-mlir check-torch-mlir-python
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

## PyTorch source builds and custom PyTorch versions

Torch-MLIR by default builds with the latest nightly PyTorch version. This can be toggled to build from latest PyTorch source with

```
-DTORCH_MLIR_USE_INSTALLED_PYTORCH=OFF
-DTORCH_MLIR_SRC_PYTORCH_REPO=vivekkhandelwal1/pytorch # Optional. Github path. Defaults to pytorch/pytorch
-DTORCH_MLIR_SRC_PYTORCH_BRANCH=master # Optional. Defaults to PyTorch's main branch
```

## Updating the LLVM and MLIR-HLO submodules

Torch-MLIR depends on `llvm-project` (which contains, among other things,
upstream MLIR) and `stablehlo`, both of which are submodules in the `externals/`
directory. We aim to update these at least weekly to bring in the latest
features and spread out over time the effort of updating our code for MLIR API
breakages.

### Which LLVM commit should I pick?

NOTE: This section is in flux. Specifically, the `mlir-hlo` dep has been
dropped and the project is running off of a `stablehlo` fork which can be
patched for certain OS combinations. As of 2023-09-12, stellaraccident@
is massaging this situation. Please reach out for advice updating.

Since downstream projects may want to build Torch-MLIR (and thus LLVM and
MLIR-HLO) in various configurations (Release versus Debug builds; on Linux,
Windows, or macOS; possibly with Clang, LLD, and LLDB enabled), it is crucial to
pick LLVM commits that pass tests for all combinations of these configurations.

So every week, we track the so-called _green_ commit (i.e. the LLVM commit which
works with all of the above configurations) in Issue
https://github.com/llvm/torch-mlir/issues/1178.  In addition to increasing our
confidence that the resulting update will not break downstream projects, basing
our submodule updates on these green commits also helps us stay in sync with
LLVM updates in other projects like ONNX-MLIR and MLIR-HLO. The person
responsible for the update each week is listed [here](https://github.com/llvm/torch-mlir/wiki/Weekly-LLVM-Update).

### What is the update process?

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
5. **Update Abstract Interpretation Library**: Run
   `build_tools/update_abstract_interp_lib.sh`.  This is sometimes needed
   because upstream changes can affect canonicalization and other minor details
   of the IR in the abstract interpretation library. See
   [docs/abstract_interp_lib.md](abstract_interp_lib.md) for more details
   on the abstract interpretation library.


Here are some examples of PRs updating the LLVM and MLIR-HLO submodules:

- https://github.com/llvm/torch-mlir/pull/1180
- https://github.com/llvm/torch-mlir/pull/1229

## Enabling Address Sanitizer (ASan)

To enable ASAN, pass `-DLLVM_USE_SANITIZER=Address` to CMake. This should "just
work" with all C++ tools like `torch-mlir-opt`. When running a Python script
such as through `./projects/pt1/tools/e2e_test.sh`, you will need to do:

```
LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)" ./projects/pt1/tools/e2e_test.sh -s
# See instructions here for how to get the libasan path for GCC:
# https://stackoverflow.com/questions/48833176/get-location-of-libasan-from-gcc-clang
```

TODO: Add ASan docs for LTC.

## Other docs

- GitHub wiki: https://github.com/llvm/torch-mlir/wiki
- Of particular interest in the [How to add end-to-end support for new Torch ops](https://github.com/llvm/torch-mlir/wiki/Torch-ops-E2E-implementation) doc.
