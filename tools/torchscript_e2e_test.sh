#!/bin/bash
set -euo pipefail

src_dir="$(realpath $(dirname $0)/..)"

cd "$src_dir"
source .env
python -m frontends.pytorch.e2e_testing.torchscript.main "$@"
