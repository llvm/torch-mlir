# npcomp - An aspirational MLIR based numpy compiler

This is a research prototype of MLIR dialects for representing
numpy programs, and a set of reference tracing/compiler tools.
The primary purpose at this point is to establish a solid modeling
of restricted Python programs and Numpy based computations in MLIR.
While this project will provide some reference implementations to prove
the design, the intention is to align this with the broader set of
tools that exist at this level of abstraction.

## Design Notes

As I work through things, I've been jotting down some design notes:

* [Type Extraction - April 15, 2020](https://gist.github.com/stellaraccident/ec1ab0f633cfca0a05866fd77705b4e4)
* [Ufunc modeling Part 1 - April 29, 2020](https://gist.github.com/stellaraccident/4fcd2a24a66b6588f92b22b2b8ab974f)
* [Array funcs and op granularity - May 5, 2020](https://gist.github.com/stellaraccident/2c11652cfdee1457921bc7c98807b462)

## Quick start

```
LLVM_VERSION=10
export CC=clang-$LLVM_VERSION
export CXX=clang++-$LLVM_VERSION
export LDFLAGS=-fuse-ld=$(which ld.lld-$LLVM_VERSION)
export LLVM_SRC_DIR=/path/to/llvm-project

# Check out last known good commit.
(cd $LLVM_SRC_DIR && git checkout 3af85fa8f06220b43f03f26de216a67be4568fe7)

./tools/install_mlir.sh
./tools/cmake_configure.sh


# ./tools/test_all.sh runs all of these commands.
cd build
ninja
ninja check-npcomp-opt
# Note: currently, python tests run separately
./python/run_tests.py
```

## Interactive Use

The cmake configuration populates symlinks in the `build/python` directory
mirroring the source layout. This allows edit-run without rebuilding (unless
if files are added/removed).

Configuring the `PYTHONPATH` should be sufficient to run any interactive
tooling (`python3`, Jupyter/Colab, etc).

```shell
export PYTHONPATH="$(realpath build/python):$(realpath build/python_native)"
```

The `run_tests.py` script is special in that it sets up the PYTHONPATH
correctly when run.

Note that running the `cmake_configure.sh` script will also output a `.env`
file in the workspace folder with the correct PYTHONPATH set. This allows
tools like VSCode to work by default for debugging.

### Things to look at:

* `python/npcomp/tracing/mlir_trace_test.py` : Simple test case of tracing a function to an MLIR module.

Notes:

* Python sources are symlinked to the output directory at configure time.
  Adding sources will require a reconfigure. Editing should not.
* It is a very common issue to have both python 2.7 (aka. "python") and python
  3.x (aka. "python3") on a system at a time (and we can only hope that one 
  day this ends). Since the native library at development time binds to a
  specific version, if you try to run with a different python, you will get
  an error about the "native" module not being found.

