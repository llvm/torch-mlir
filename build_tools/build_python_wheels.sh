#!/bin/bash
set -e

if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi
version="$("$PYTHON" --version)"
echo "Using python: $PYTHON (version $version)"

repo_root="$(cd $(dirname $0)/.. && pwd)"
wheelhouse="$repo_root/wheelhouse"
package_test_venv="$wheelhouse/package-test.venv"
mkdir -p $wheelhouse
cd $wheelhouse

echo "---- BUILDING torch-mlir ----"
CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
$PYTHON "${repo_root}/setup.py" bdist_wheel --dist-dir "$wheelhouse" -v

# Smoke test: create a venv, install the package, and run an example.

echo "---- CREATING VENV ----"
python -m venv "$package_test_venv"
VENV_PYTHON="$package_test_venv/bin/python"

echo "---- INSTALLING torch ----"
$VENV_PYTHON -m pip install --pre torch torchvision pybind11 -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
echo "---- INSTALLING other deps for smoke test ----"
$VENV_PYTHON -m pip install requests pillow
echo "---- INSTALLING torch-mlir ----"
$VENV_PYTHON -m pip install -f "$wheelhouse" --force-reinstall torch_mlir

echo "---- RUNNING SMOKE TEST ----"
$VENV_PYTHON "$repo_root/examples/torchscript_resnet18_e2e.py"
