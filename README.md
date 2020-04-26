# npcomp - An aspirational MLIR based numpy compiler

## Scratch-pad of build configurations that have worked

### VSCode settings for configuring CMake

```json
"cmake.configureArgs": [
  "-DLLVM_TARGETS_TO_BUILD=X86",
  "-DLLVM_ENABLE_PROJECTS=mlir;npcomp",
  "-DPYTHON_EXECUTABLE=/bin/python3",
  "-DLLVM_EXTERNAL_PROJECTS=npcomp",
  "-DLLVM_ENABLE_ASSERTIONS:BOOL=ON"
]
```

### Installing pybind11

The native extension relies on pybind11. In a perfect world, this could just
be installed with your system package manager. However, at least on
Ubuntu Disco, the system package installed with broken cmake files.

I built/installed from pybind11 head without issue and put it in /usr/local.
There are better ways to do this.

### Building the python native library

```shell
# From the build directory
ninja NPCOMPNativePyExt
# Outputs to tools/npcomp/python/npcomp/native...so
export PYTHONPATH=$(pwd)/tools/npcomp/python
python3 -m npcomp.smoketest
```

Notes:

* Python sources are symlinked to the output directory at configure time.
  Adding sources will require a reconfigure. Editing should not.
* It is a very common issue to have both python 2.7 (aka. "python") and python
  3.x (aka. "python3") on a system at a time (and we can only hope that one 
  day this ends). Since the native library at development time binds to a
  specific version, if you try to run with a different python, you will get
  an error about the "native" module not being found.
