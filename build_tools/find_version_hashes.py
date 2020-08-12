#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find common version hashes for dependent projects.

Sample usage:
  ./build_tools/find_version_hashes.py --iree_dir=${IREE_DIR}

This script will fetch dependent projects and seek back over the last
--revision-depth commits against their respective version files in order to
find common revisions of each that share a same common LLVM hash, reporting
all such hashes.

Note that this procedure is not guaranteed to work or produce a recent
version. It has a reasonable probability of working since the non-LLVM
dependencies are published by Google at regular intervals and common LLVM
commits.

In general, unless if the versions found by this script are too old, integrating
at it's recommendation will increase the probability that dependencies are
actually mutually compatible with each other and make for an easier time
upgrading. It is experimental and subject to change.
"""

import argparse
import collections
import os
import subprocess
import sys

TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_argument_parser():
  parser = argparse.ArgumentParser(
      prog="find_version_hashes.py",
      description="Finds common version hashes for sub-projects")
  parser.add_argument("--llvm-dir",
                      help="Directory of the llvm-project repo",
                      default="external/llvm-project")
  parser.add_argument("--mlir-hlo-dir",
                      help="Directory of the MLIR HLO project checkout",
                      default=os.path.join(TOP_DIR, "external", "mlir-hlo"))
  parser.add_argument("--iree-dir",
                      help="Directory of the IREE project checkout (optional)",
                      default=None)
  parser.add_argument(
      "--revision-depth",
      type=int,
      help="The number of revisions to search back for a common join",
      default=50)
  parser.add_argument("--no-fetch",
                      help="Do not fetch child repositories",
                      action="store_true")
  return parser


ChildRevisionMap = collections.namedtuple(
    "ChildRevisionMap", "parent_revision,parent_date,child_revision,child_date")

ParentChildJoin = collections.namedtuple(
    "ParentChildJoin", "parent_revision,parent_date,child_revision_maps")


def main(args):
  if not args.no_fetch:
    fetch(args.llvm_dir)
  llvm_revision_maps = {}
  if args.mlir_hlo_dir:
    llvm_revision_maps["mhlo"] = get_mhlo_llvm_history(args)
  if args.iree_dir:
    llvm_revision_maps["iree"] = get_iree_llvm_history(args)

  # Join the LLVM revision.
  join_results = join_child_revision_maps(llvm_revision_maps)
  if not join_results:
    print("No common LLVM version found (TODO print a better report)s.")
    print(llvm_revision_maps)
    sys.exit(1)

  # Report.
  print("COMMON LLVM REVISION: {} (at {})".format(join_results.parent_revision,
                                                  join_results.parent_date))
  for child_key, child_revision_map in join_results.child_revision_maps.items():
    print("  - {}: {} (at {})".format(child_key,
                                      child_revision_map.child_revision,
                                      child_revision_map.child_date))


def join_child_revision_maps(revision_maps):
  """Joins dicts of child_key -> [ChildRevisionMap].

  Returns:
    Return ParentChildJoin or None if no revisions found.
  """
  parent_revision_dates = dict()  # Dates of each parent revision.
  parent_revisions = dict()  # Of parent_revision -> count of agreements.
  for child_key, child_maps in revision_maps.items():
    for child_map in child_maps:
      parent_revision_dates[child_map.parent_revision] = child_map.parent_date
      count = parent_revisions.get(child_map.parent_revision)
      parent_revisions[child_map.parent_revision] = (
          (0 if count is None else count) + 1)

  def build_child_map(parent_revision):
    child_map = dict()
    for child_key, child_revision_map in revision_maps.items():
      for single_child_revision_map in child_revision_map:
        if single_child_revision_map.parent_revision == parent_revision:
          child_map[child_key] = single_child_revision_map
          break
    return child_map

  # Find the most recent parent commit where all children agree.
  expected_children = len(revision_maps)
  for parent_revision, count in parent_revisions.items():
    if count == expected_children:
      # Found it!
      return ParentChildJoin(parent_revision,
                             parent_revision_dates[parent_revision],
                             build_child_map(parent_revision))
  return None


def get_mhlo_llvm_history(args):
  """Mlir-hlo stores its llvm commit hash in a text file which is parsed.

  Returns:
    List of ChildRevisionMap.
  """
  if not args.no_fetch:
    fetch(args.mlir_hlo_dir)
  mlir_hlo_revisions = get_file_revisions(args.mlir_hlo_dir,
                                          "build_tools/llvm_version.txt",
                                          revision_depth=args.revision_depth)
  # Re-arrange into (llvm_revision, llvm_date, child_revision, child_date)
  llvm_history = []
  for child_revision, child_date, contents in mlir_hlo_revisions:
    llvm_revision = contents.decode("UTF-8").strip()
    llvm_date = get_commit_date(args.llvm_dir, llvm_revision)
    llvm_history.append(
        ChildRevisionMap(llvm_revision, llvm_date, child_revision, child_date))
  return llvm_history


def get_iree_llvm_history(args):
  """Gets the IREE LLVM history by parsing the SUBMODULE_VERSIONS file.

  Returns:
    List of ChildRevisionMap.
  """
  if not args.no_fetch:
    fetch(args.iree_dir)
  iree_revisions = get_file_revisions(args.iree_dir,
                                      "SUBMODULE_VERSIONS",
                                      revision_depth=args.revision_depth)

  def get_llvm_revision(submodule_versions):
    # Each line is "hash path/to/module"
    for line in submodule_versions.decode("UTF-8").strip().splitlines():
      revision, path = line.split(" ", maxsplit=1)
      if path == "third_party/llvm-project":
        return revision
    return None

  llvm_history = []
  for child_revision, child_date, contents in iree_revisions:
    llvm_revision = get_llvm_revision(contents)
    if llvm_revision is None:
      print(
          "Could not find llvm-project revision in IREE SUBMODULE_VERSIONS:\n" +
          contents.decode("UTF-8"),
          file=sys.stderr)
    llvm_date = get_commit_date(args.llvm_dir, llvm_revision)
    llvm_history.append(
        ChildRevisionMap(llvm_revision, llvm_date, child_revision, child_date))
  return llvm_history


def get_commit_date(repo_path, revision):
  """Gets the date of a commit."""
  return subprocess_call(
      ["git", "log", "-n", "1", "--pretty=format:%ci", revision],
      cwd=repo_path,
      capture_output=True).decode("UTF-8").strip()


def get_file_revisions(repo_path, file_path, revision_depth):
  """Gets the file contents at the last `revision-depth` commits.

  Returns:
    A tuple of (revision, date, contents).
  """
  revisions = subprocess_call([
      "git", "log", "--pretty=format:%H %ci", "-n",
      str(revision_depth), "origin/HEAD", "--", file_path
  ],
                              cwd=repo_path,
                              capture_output=True).decode("UTF-8").splitlines()
  # Split on space.
  revisions = [r.split(" ", maxsplit=1) for r in revisions]

  # Generate the revision tuple (revision, date, contents).
  def show_contents(revision):
    return subprocess_call(["git", "show", "{}:{}".format(revision, file_path)],
                           cwd=repo_path,
                           capture_output=True)

  revision_contents = [
      (revision, date, show_contents(revision)) for revision, date in revisions
  ]
  return revision_contents


def fetch(repo_path):
  print("Fetching", repo_path, "...", file=sys.stderr)
  subprocess_call(["git", "fetch", "--recurse-submodules=no"], cwd=repo_path)


def subprocess_call(args, cwd, capture_output=False, **kwargs):
  """Calls a subprocess, possibly capturing output."""
  try:
    if capture_output:
      return subprocess.check_output(args, cwd=cwd, **kwargs)
    else:
      return subprocess.check_call(args, cwd=cwd, **kwargs)
  except subprocess.CalledProcessError:
    print("ERROR executing subprocess (from {}):\n  {}".format(
        cwd, " ".join(args)),
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main(create_argument_parser().parse_args(sys.argv[1:]))
