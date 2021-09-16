#!/bin/bash

# Script for invoking all test tests. It also prints out a banner in case
# of success.
# TODO: I think we can just remove this.

set -euo pipefail
td="$(realpath $(dirname $0)/..)"

cd $td/build

ninja
ninja check-npcomp-all

echo
echo "========"
echo "ALL PASS"
echo "========"
