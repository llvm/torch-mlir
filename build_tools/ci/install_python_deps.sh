#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
torch_version="${1:-unknown}"

echo "::group::installing llvm python deps"
python -m pip install --no-cache-dir -r $repo_root/externals/llvm-project/mlir/python/requirements.txt
echo "::endgroup::"

case $torch_version in
  nightly)
    echo "::group::installing nightly torch"
    python3 -m pip install --no-cache-dir -r $repo_root/requirements.txt
    python3 -m pip install --no-cache-dir -r $repo_root/torchvision-requirements.txt
    echo "::endgroup::"
    ;;
  stable)
    echo "::group::installing stable torch"
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install --no-cache-dir -r $repo_root/build-requirements.txt
    echo "::endgroup::"
    ;;
  *)
    echo "Unrecognized torch version '$torch_version' (specify 'nightly' or 'stable' with cl arg)"
    exit 1
    ;;
esac

echo "::group::installing test requirements"
python -m pip install --no-cache-dir -r $repo_root/test-requirements.txt
echo "::endgroup::"
