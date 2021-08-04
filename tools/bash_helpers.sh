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
  # Also produce a reproducer by default.
  "${build_dir}/bin/npcomp-opt" \
    "-pass-pipeline-crash-reproducer=/tmp/reproducer.mlir" \
    "$@"
}

refback-run() {
  # Helper for building and invoking refback-run.
  #
  # This also automatically builds and adds the npcomp runtime shared
  # library.
  #
  # Usage:
  # $ refback-run <regular refback-run options>
  ninja -C "$build_dir" refback-run NPCOMPCompilerRuntimeShlib 1>&2 || return 1
  "$build_dir/bin/refback-run" \
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
