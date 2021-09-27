#!/bin/bash
set -euo pipefail

# Check that only two arugments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <venv_dir> <serialized_test_dir>"
    echo 'Description:
    venv_dir: directory to put the Python venv used for generating the serialized tests
    serialized_test_dir: directory to write the generated serialized tests to
'
    exit 1
fi

venv_dir=$1
serialized_test_dir=$2
here="$(realpath $(dirname $0))"
torch_mlir_src_root="$here/../../"

mkdir -p $venv_dir
mkdir -p $serialized_test_dir
python3 -m venv $venv_dir
source $venv_dir/bin/activate
python3 -m pip install fairseq fvcore sacremoses subword-nmt

cd "$torch_mlir_src_root"
export PYTHONPATH=${PYTHONPATH-}
source "$torch_mlir_src_root/.env"
python3 -m build_tools.torchscript_e2e_heavydep_tests.main --output_dir=$serialized_test_dir
