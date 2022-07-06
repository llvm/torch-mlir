#!/bin/bash
set -eu -o pipefail

if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi
version="$("$PYTHON" --version)"
echo "Using python: $PYTHON (version $version)"

repo_root="$(cd "$(dirname "$0")"/.. && pwd)"
wheelhouse="$repo_root/wheelhouse"
package_test_venv="$wheelhouse/package-test.venv"
mkdir -p "$wheelhouse"
cd "$wheelhouse"

echo "---- BUILDING torch-mlir ----"
CMAKE_GENERATOR=Ninja \
$PYTHON "${repo_root}/setup.py" bdist_wheel --dist-dir "$wheelhouse" -v

# Smoke test: create a venv, install the package, and run an example.

echo "---- CREATING VENV ----"
python -m venv "$package_test_venv"
VENV_PYTHON="$package_test_venv/bin/python"

# Install the Torch-MLIR package.
# Note that we also need to pass in the `-r requirements.txt` here to pick up
# the right --find-links flag for the nightly PyTorch wheel registry.
echo "---- INSTALLING torch-mlir and dependencies ----"
$VENV_PYTHON -m pip install -f "$wheelhouse" --force-reinstall torch_mlir -r "${repo_root}/requirements.txt"
echo "---- INSTALLING other deps for smoke test ----"
$VENV_PYTHON -m pip install requests pillow

echo "---- RUNNING SMOKE TEST ----"
$VENV_PYTHON "$repo_root/examples/torchscript_resnet18.py"
