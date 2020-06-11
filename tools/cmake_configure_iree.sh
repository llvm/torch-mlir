#!/bin/bash
# Experimental configure script that includes IREE.
set -e

td="$(realpath $(dirname $0))"
iree_dir="$(realpath "$td/../../iree")"
if ! [ -d "$iree_dir" ]; then
  echo "Could not find IREE src dir: $iree_dir"
  exit 1
fi

"$td/cmake_configure.sh" \
  -DNPCOMP_ENABLE_IREE=1 \
  "-DNPCOMP_IREE_SRCDIR=$iree_dir" \
  "$@"
