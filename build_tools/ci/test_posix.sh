#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
torch_version="${1:-unknown}"

export PYTHONPATH="$repo_root/build/tools/torch-mlir/python_packages/torch_mlir:$repo_root/projects/pt1"

echo "::group::Run ONNX e2e integration tests"
python -m e2e_testing.main --config=onnx -v --filter AtenNonzero1DModule_one_nonzero # fail
# python -m e2e_testing.main --config=linalg -v --filter NonzeroDecomposeModule_basic # Passed: 1

# python -m e2e_testing.main --config=onnx -v --filter NonzeroDecomposeModule_basic # Failed: 1

# python -m e2e_testing.main --config=linalg -v --filter NonzeroFlattenDynamicModule # Passed: 1

# python -m e2e_testing.main --config=onnx -v --filter ScatterAddDynamicModule_basic #  Passed: 1

# python -m e2e_testing.main --config=onnx -v --filter NonzeroCatModule # Passed: 1
# python -m e2e_testing.main --config=linalg -v --filter NonzeroCatModule # Failed: 1
# tensor with unknown dtype "torch.aten.cat"(%31, %4) : (!torch.list<vtensor>, !torch.int) -> !torch.vtensor<[1],unk>

# python -m e2e_testing.main --config=linalg -v --filter NonzeroCatModule # Failed: 1

# python -m e2e_testing.main --config=linalg -v --filter NonzeroCumsumModule
# python -m e2e_testing.main --config=onnx -v --filter NonzeroCumsumModule # pass
# python -m e2e_testing.main --config=onnx -v --filter NonzeroCumsumBoolModule # pass in torch-mlir, failed in iree

# python -m e2e_testing.main --config=onnx -v --filter NonzeroLongModule
echo "::endgroup::"

# case $torch_version in
#   nightly)
#     # Failing with: NotImplementedError:
#     #   Could not run 'aten::empty.memory_format' with arguments from the 'Lazy' backend.
#     # As of 2024-01-07
#     # echo "::group::Run Lazy Tensor Core e2e integration tests"
#     # python -m e2e_testing.main --config=lazy_tensor_core -v
#     # echo "::endgroup::"

#     # TODO: Need to verify in the stable version
#     echo "::group::Run FxImporter e2e integration tests"
#     python -m e2e_testing.main --config=fx_importer -v
#     echo "::endgroup::"

#     # TODO: Need to verify in the stable version
#     echo "::group::Run FxImporter2Stablehlo e2e integration tests"
#     python -m e2e_testing.main --config=fx_importer_stablehlo -v
#     echo "::endgroup::"
#     ;;
#   stable)
#     ;;
#   *)
#     echo "Unrecognized torch version '$torch_version' (specify 'nightly' or 'stable' with cl arg)"
#     exit 1
#     ;;
# esac
