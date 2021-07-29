#!/bin/bash
set -e

if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi
version="$("$PYTHON" --version)"
echo "Using python: $PYTHON (version $version)"

repo_root="$(cd $(dirname $0)/.. && pwd)"
wheelhouse="$repo_root/wheelhouse"
mkdir -p $wheelhouse
cd $wheelhouse

echo "---- BUILDING npcomp-core ----"
CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
$PYTHON -m pip wheel "${repo_root}" \
  --use-feature=in-tree-build \
  -w "$wheelhouse" -v

echo "---- INSTALLING npcomp-core ----"
$PYTHON -m pip install -f "$wheelhouse" --force-reinstall npcomp-core

echo "---- BUILDING npcomp-torch ----"
$PYTHON -m pip wheel "${repo_root}/frontends/pytorch" \
  --use-feature=in-tree-build \
  -w "$wheelhouse" -v

echo "---- INSTALLING npcomp-torch ----"
$PYTHON -m pip install -f "$wheelhouse" --force-reinstall npcomp-torch

echo "---- QUICK SMOKE TEST ----"
$PYTHON $repo_root/frontends/pytorch/test/torchscript_e2e_test/basic.py
