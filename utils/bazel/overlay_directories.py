#!/bin/python3

# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Overlays two directories into a target directory using symlinks.

Tries to minimize the number of symlinks created (that is, does not symlink
every single file). Symlinks every file in the overlay directory. Only symlinks
individual files in the source directory if their parent directory is also
contained in the overlay directory tree.
"""

import argparse
import errno
import os
import sys


def check_python_version():
    """Checks if the Python version is at least 3."""
    if sys.version_info[0] < 3:
        raise RuntimeError(
            "Must be invoked with a python 3 interpreter but was %s" %
            sys.executable)


def check_dir_exists(path):
    """Checks if the directory exists."""
    if not os.path.isdir(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
    Overlays two directories into a target directory using symlinks.

    Tries to minimize the number of symlinks created (that is, does not symlink
    every single file). Symlinks every file in the overlay directory. Only
    symlinks individual files in the source directory if their parent directory
    is also contained in the overlay directory tree.
    """)
    parser.add_argument(
        "--src",
        required=True,
        help="Directory that contains most of the content to symlink.")
    parser.add_argument(
        "--overlay",
        required=True,
        help="Directory to overlay on top of the source directory.")
    parser.add_argument(
        "--target",
        required=True,
        help="Directory in which to place the fused symlink directories.")

    args = parser.parse_args()

    check_dir_exists(args.target)
    check_dir_exists(args.overlay)
    check_dir_exists(args.src)

    return args


def symlink_abs(from_path, to_path):
    """Creates an absolute symlink from 'from_path' to 'to_path'."""
    if not os.path.exists(to_path):
        os.symlink(os.path.abspath(from_path), os.path.abspath(to_path))


def main(args):
    """Main function to overlay directories using symlinks."""
    for root, dirs, files in os.walk(args.overlay):
        rel_root = os.path.relpath(root, start=args.overlay)
        if rel_root != ".":
            os.mkdir(os.path.join(args.target, rel_root))

        for file in files:
            relpath = os.path.join(rel_root, file)
            symlink_abs(os.path.join(args.overlay, relpath),
                        os.path.join(args.target, relpath))

        for src_entry in os.listdir(os.path.join(args.src, rel_root)):
            if src_entry not in dirs:
                relpath = os.path.join(rel_root, src_entry)
                symlink_abs(os.path.join(args.src, relpath),
                            os.path.join(args.target, relpath))


if __name__ == "__main__":
    check_python_version()
    main(parse_arguments())
