# No shebang, please `source` this file!
# For example, add this to your `.bashrc`:
# ```
# source $WHERE_YOU_CHECKED_OUT_NPCOMP/tools/bash_helpers.sh
# ```

td="$(realpath $(dirname "${BASH_SOURCE[0]}")/..)"
build_dir="$td/build"

npcomp-opt() {
  # Helper for building and invoking npcomp-opt.
  # Usage:
  # $ npcomp-opt <regular npcomp-opt options>
  ninja -C "$build_dir" npcomp-opt 1>&2 || return 1
  "${build_dir}/tools/npcomp-opt/npcomp-opt" "$@"
}

npcomp-run-mlir() {
  # Helper for building and invoking npcomp-run-mlir.
  #
  # This also automatically builds and adds the npcomp runtime shared
  # library.
  #
  # Usage:
  # $ npcomp-run-mlir <regular npcomp-run-mlir options>
  ninja -C "$build_dir" npcomp-run-mlir NPCOMPCompilerRuntimeShlib 1>&2 || return 1
  $build_dir/tools/npcomp-run-mlir/npcomp-run-mlir \
    -shared-libs="${build_dir}/lib/libNPCOMPCompilerRuntimeShlib.so" "$@"
}

mnist-playground() {
  # Helper for building and invoking mnist-playground
  #
  # This also automatically builds and adds the npcomp runtime shared
  # library.
  #
  # Usage:
  # $ mnist-playground <regular mnist-playground options>
  ninja -C "$build_dir" mnist-playground NPCOMPCompilerRuntimeShlib 1>&2 || return 1
  $build_dir/tools/mnist-playground/mnist-playground \
    -shared-libs="${build_dir}/lib/libNPCOMPCompilerRuntimeShlib.so" "$@"
}

# Go to the root of your npcomp checkout.
alias npd="cd $td"
# Handy so that `npctest -v` runs lit with -v and thus prints out errors,
# which `check-npcomp` does not.
npctest() {
  (npd && tools/run_lit.sh test "$@")
}
npctall() {
  (npd && build_tools/test_all.sh "$@")
}

# Don't autocomplete `install-mlir` which conflicts with `include` in our
# typical build setup.
#
# See here for more info about the incantation:
# https://stackoverflow.com/a/34272881
# https://superuser.com/q/253068
export FIGNORE=$FIGNORE:nstall-mlir

export PYTHONPATH="$(realpath ${build_dir}/python):$(realpath ${build_dir}/python_native):$(realpath ${build_dir}/iree/bindings/python)"

