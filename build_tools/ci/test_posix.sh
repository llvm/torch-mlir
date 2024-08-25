#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
torch_version="${1:-unknown}"

export PYTHONPATH="$repo_root/build/tools/torch-mlir/python_packages/torch_mlir:$repo_root/projects/pt1"

case $torch_version in
  nightly)
    # Failing with: NotImplementedError:
    #   Could not run 'aten::empty.memory_format' with arguments from the 'Lazy' backend.
    # As of 2024-01-07
    # echo "::group::Run Lazy Tensor Core e2e integration tests"
    # python -m e2e_testing.main --config=lazy_tensor_core -v
    # echo "::endgroup::"

    # TODO: Need to verify in the stable version
    echo "::group::Run FxImporter e2e integration tests"
    python -m e2e_testing.main --config=fx_importer -v
    echo "::endgroup::"
    ;;
  stable)
    ;;
  *)
    echo "Unrecognized torch version '$torch_version' (specify 'nightly' or 'stable' with cl arg)"
    exit 1
    ;;
esac
