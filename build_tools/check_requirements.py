# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import sys
import tomllib


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    requirements_path = os.path.join(
        project_root,
        "externals",
        "llvm-project",
        "mlir",
        "python",
        "requirements.txt",
    )

    if not os.path.exists(requirements_path):
        print(
            f"Error: {requirements_path} not found. Submodules might not be"
            " checked out.",
            file=sys.stderr,
        )
        return 1

    with open(pyproject_path, "rb") as f:
        try:
            pyproject = tomllib.load(f)
            build_requires = pyproject["build-system"]["requires"]
        except Exception as e:
            print(f"Error parsing {pyproject_path}: {e}", file=sys.stderr)
            return 1

    with open(requirements_path, "r") as f:
        mlir_requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    missing = [req for req in mlir_requirements if req not in build_requires]
    if missing:
        print(
            "Error: The following MLIR requirements are missing in"
            " pyproject.toml:\n" + "\n".join(f"  - {m}" for m in missing),
            file=sys.stderr,
        )
        return 1

    print("Requirements are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
