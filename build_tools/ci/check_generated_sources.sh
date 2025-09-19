#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"

function _check_file_not_changed_by() {
  # _check_file_not_changed_by <cmd> <file>
  cmd="$1"
  file="$2"
  file_backup="$PWD/$(basename $file)"
  file_new="$PWD/$(basename $file).new"
  # Save the original file.
  cp "$file" "$file_backup"
  # Run the command to regenerate it.
  "$1" || return 1
  # Save the new generated file.
  cp "$file" "$file_new"
  # Restore the original file. We want this function to not change the user's
  # working tree state.
  mv "$file_backup" "$file"
  # We use git-diff as "just a diff program" (no SCM stuff) because it has
  # nicer output than regular `diff`.
  if ! git diff --no-index --quiet "$file" "$file_new"; then
    echo "#######################################################"
    echo "Generated file '${file}' is not up to date (see diff below)"
    echo ">>> Please run '${cmd}' to update it <<<"
    echo "#######################################################"
    git diff --no-index --color=always "$file" "$file_new"
    # TODO: Is there a better cleanup strategy that doesn't require duplicating
    # this inside and outside the `if`?
    rm "$file_new"
    return 1
  fi
  rm "$file_new"
}

echo "::group:: Check that update_abstract_interp_lib.sh has been run"
_check_file_not_changed_by $repo_root/build_tools/update_abstract_interp_lib.sh $repo_root/lib/Dialect/Torch/Transforms/AbstractInterpLibrary.cpp
echo "::endgroup::"

echo "::group:: Check that update_torch_ods.sh has been run"
_check_file_not_changed_by $repo_root/build_tools/update_torch_ods.sh $repo_root/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
echo "::endgroup::"
