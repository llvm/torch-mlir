#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
torch_version="${1:-unknown}"

export PYTHONPATH="$repo_root/build/tools/torch-mlir/python_packages/torch_mlir:$repo_root/projects/pt1"

echo "::group::Run Linalg e2e integration tests"
python -m e2e_testing.main --config=linalg -v
echo "::endgroup::"

echo "::group::Run make_fx + TOSA e2e integration tests"
python -m e2e_testing.main --config=make_fx_tosa -v
echo "::endgroup::"

echo "::group::Run TOSA e2e integration tests"
python -m e2e_testing.main --config=tosa -v
echo "::endgroup::"

echo "::group::Run Stablehlo e2e integration tests"
python -m e2e_testing.main --config=stablehlo -v
echo "::endgroup::"

echo "::group::Run ONNX e2e integration tests"
python -m e2e_testing.main --config=onnx -v
echo "::endgroup::"

case $torch_version in
  nightly)
    # Failing with: NotImplementedError: 
    #   Could not run 'aten::empty.memory_format' with arguments from the 'Lazy' backend.
    # As of 2024-01-07
    # echo "::group::Run Lazy Tensor Core e2e integration tests"
    # python -m e2e_testing.main --config=lazy_tensor_core -v
    # echo "::endgroup::"

    # TODO: There is one failing test in this group on stable. It could
    # be xfailed vs excluding entirely.
    echo "::group::Run TorchDynamo e2e integration tests"
    python -m e2e_testing.main --config=torchdynamo -v
    echo "::endgroup::"
    ;;
  stable)
    ;;
  *)
    echo "Unrecognized torch version '$torch_version' (specify 'nightly' or 'stable' with cl arg)"
    exit 1
    ;;
esac
