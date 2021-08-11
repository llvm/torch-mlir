#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <iree_src_root>"
    echo 'Description:
    iree_src_root: root directory of IREE source checkout
'
    exit 1
fi

npcomp_src_root="$(realpath $(dirname $0)/..)"
iree_src_root=$1

rm -rf "${npcomp_src_root}/external/iree-dialects"
cp -a "${iree_src_root}/llvm-external-projects/iree-dialects" "${npcomp_src_root}/external"
