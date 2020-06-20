# NPComp - An aspirational MLIR based numpy compiler

NPComp aims to be an idiomatic subset of the Python language, suitable for
extracting isolated, statically typed programs from a running Python session.
It is inspired by many projects that have come before it, including:

* PyPy/RPython
* Numba
* Pythran
* TorchScript
* Autograph

As the name implies, NPComp also seeks to provide compiler-backed support
for Numpy APIs.

The project spawned out of both [LLVM's MLIR project](https://mlir.llvm.org/)
and [The IREE Project](https://github.com/google/iree) and seeks to use the
MLIR and IREE tooling to enable progressive lowering of high level compute
dominant sub-programs in a way that preserves high level semantic information
that is expected to be useful for exploiting parallelism, generating high
performance code, and enabling portability and deployment to a range of
devices. Some of these goals overlap with existing projects, and to a first
approximation, the experiment with NPComp is to determine whether rebasing
on the MLIR tooling and ML backends like IREE produce a lift.

Before getting too excited, keep in mind that this project *barely* exists: it
is very new and doesn't do anything useful yet :) We are using it as a testing
ground for some new ideas and infrastructure improvement, and depending on
how things turn out, may end up carrying it forward or breaking it up for
parts.

See the [features doc](docs/features.md) for a semi-curated status of what is
implemented.

## Architecture

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
* [_npcomp native module](python_native)
* C++ [include](include) and [lib](lib) trees, following LLVM/MLIR conventions
* LIT testing trees:
  * [test](test): Lit/FileCheck tests covering core MLIR based infra
  * [pytest/Compiler](pytest/Compiler): Lit test suite that drive the compiler 
    infra from Python
  * [backend_test](backend_test): Lit test suites conditionally enabled for
    each backend
* [tools](tools): Scripts and binaries (npcomp-opt, npcomp-run-mlir, etc)

## Quick start

```
LLVM_VERSION=10
export CC=clang-$LLVM_VERSION
export CXX=clang++-$LLVM_VERSION
export LDFLAGS=-fuse-ld=$(which ld.lld-$LLVM_VERSION)
export LLVM_SRC_DIR=/path/to/llvm-project

# Check out last known good commit.
LLVM_COMMIT="$(cat ./built_tools/llvm.version)"
(cd $LLVM_SRC_DIR && git checkout $LLVM_COMMIT)

./build_tools/install_mlir.sh
./build_tools/cmake_configure.sh

# Build and run tests
# ./build_tools/test_all.sh runs all of these commands.
cd build
ninja
ninja check-npcomp
# Note: currently, some python tests run separately
./python/run_tests.py

# Setup PYTHONPATH for interactive use
export PYTHONPATH="$(realpath build/python):$(realpath build/python_native):$(realpath build/iree/bindings/python)"
```

## Interactive Use

The cmake configuration populates symlinks in the `build/python` directory
mirroring the source layout. This allows edit-run without rebuilding (unless
if files are added/removed).

Configuring the `PYTHONPATH` as above should be sufficient to run any 
interactive tooling (`python3`, Jupyter/Colab, etc).

The `run_tests.py` script is special in that it sets up the PYTHONPATH
correctly when run.

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
