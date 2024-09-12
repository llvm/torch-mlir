set -x

PYTHONPATH="$(pwd)/build/tools/torch-mlir/python_packages/torch_mlir:$(pwd)/projects/pt1" python3 -m e2e_testing.main --config="$1" $2 $3 $4
