#!/bin/bash

# Runs all the tests we are aware of in mlir-npcomp.
# TODO: This should eventually all be folded into a `ninja check-npcomp`.
# Unfortunately, that seems to be a level of build engineering that we
# haven't had enough bandwidth for.
#
# The basic idea seems to be to teach `lit` how to run our Python tests.
#
# There is precendent for doing this, e.g.
# - For GoogleTest tests (i.e. "unittests"): https://github.com/llvm/llvm-project/blob/c490c5e81ac90cbf079c7cee18cd56171f1e27af/llvm/test/Unit/lit.cfg.py#L25
# - LLDB tests: https://github.com/llvm/llvm-project/blob/eb7d32e46fe184fdfcb52e0a25973e713047e305/lldb/test/API/lldbtest.py#L32
# - libcxx tests: https://github.com/llvm/llvm-project/blob/eb7d32e46fe184fdfcb52e0a25973e713047e305/libcxx/utils/libcxx/test/newformat.py#L192
#
# The `getTestsInDirectory` keyword seems to be what to search for.

set -euo pipefail
td="$(realpath $(dirname $0)/..)"

cd $td/build

ninja
ninja check-npcomp
ninja check-torch-mlir
ninja check-frontends-pytorch

echo
echo "========"
echo "ALL PASS"
echo "========"


