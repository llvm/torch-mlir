# npcomp - An aspirational MLIR based numpy compiler

This is a research prototype of MLIR dialects for representing
numpy programs, and a set of reference tracing/compiler tools.

## Design Notes

As I work through things, I've been jotting down some design notes:

* [Type Extraction - April 15, 2020](https://gist.github.com/stellaraccident/ec1ab0f633cfca0a05866fd77705b4e4)
* [Ufunc modeling Part 1 - April 29, 2020](https://gist.github.com/stellaraccident/4fcd2a24a66b6588f92b22b2b8ab974f)

## Quick start

```
export LLVM_SRC_DIR=/path/to/llvm-project
./tools/install_mlir.sh
./tools/cmake_configure.sh

cd build
ninja
./python/run_tests.py
```

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

