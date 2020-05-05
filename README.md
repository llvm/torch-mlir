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
(cd $LLVM_SRC_DIR && git checkout 26777ad7a0916ad7853aa9229bb8ec0346c68a61)

./tools/install_mlir.sh
./tools/cmake_configure.sh

cd build
ninja
ninja check-npcomp-opt
# Note: currently, python tests run separately
./python/run_tests.py
```

### Things to look at:

* `python/npcomp/tracing/mlir_trace_test.py` : Simple test case of tracing a function to an MLIR module.

### Installing pybind11

The native extension relies on pybind11. In a perfect world, this could just
be installed with your system package manager. However, at least on
some tested versions of Ubuntu, the system package installed with broken cmake 
files.

If this happens, you must install pybind11 from source.

### Building the python native library

```shell
# From the build directory
ninja NPCOMPNativePyExt
ninja check-npcomp
python3 ./python/run_tests.py

# Setup PYTHONPATH for interactive use.
export PYTHONPATH=$(pwd)/tools/npcomp/python
```

Notes:

* Python sources are symlinked to the output directory at configure time.
  Adding sources will require a reconfigure. Editing should not.
* It is a very common issue to have both python 2.7 (aka. "python") and python
  3.x (aka. "python3") on a system at a time (and we can only hope that one 
  day this ends). Since the native library at development time binds to a
  specific version, if you try to run with a different python, you will get
  an error about the "native" module not being found.

