# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import subprocess
import sys
import urllib.error
import urllib.request

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: tomli or tomllib (Python 3.11+) is required.", file=sys.stderr)
        print("Please install tomli: pip install tomli", file=sys.stderr)
        sys.exit(1)


def parse_requirements_content(content_lines):
    requirements = []
    for line in content_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def get_submodule_commit(project_root, submodule_path):
    try:
        # Run git rev-parse :path/to/submodule from project root
        result = subprocess.run(
            ["git", "rev-parse", f":{submodule_path}"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(
            f"Warning: Failed to get submodule commit for {submodule_path}: {e}",
            file=sys.stderr,
        )
        return None


def fetch_requirements_from_github(commit_hash):
    url = f"https://raw.githubusercontent.com/llvm/llvm-project/{commit_hash}/mlir/python/requirements.txt"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode("utf-8")
            return content.splitlines()
    except urllib.error.URLError as e:
        print(f"Warning: Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    requirements_path = os.path.join(
        project_root, "externals", "llvm-project", "mlir", "python", "requirements.txt"
    )

    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found.", file=sys.stderr)
        return 1

    # Read pyproject.toml
    with open(pyproject_path, "rb") as f:
        try:
            pyproject = tomllib.load(f)
        except Exception as e:
            print(f"Error parsing {pyproject_path}: {e}", file=sys.stderr)
            return 1

    try:
        build_requires = pyproject["build-system"]["requires"]
    except KeyError:
        print(
            f"Error: [build-system].requires not found in {pyproject_path}",
            file=sys.stderr,
        )
        return 1

    mlir_requirements = []
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"

    if os.path.exists(requirements_path):
        # Use local file if available
        print(f"Using local requirements file: {requirements_path}")
        with open(requirements_path, "r") as f:
            mlir_requirements = parse_requirements_content(f.readlines())
    else:
        # Try to fetch from GitHub
        print("Local requirements file not found. Trying to fetch from GitHub...")
        commit_hash = get_submodule_commit(project_root, "externals/llvm-project")
        if commit_hash:
            print(f"Submodule commit hash: {commit_hash}")
            fetched_lines = fetch_requirements_from_github(commit_hash)
            if fetched_lines:
                mlir_requirements = parse_requirements_content(fetched_lines)
                print("Successfully fetched requirements from GitHub.")

    if not mlir_requirements:
        if is_ci:
            print(
                "Error: Could not obtain MLIR requirements (local file missing "
                "and GitHub fetch failed) on CI.",
                file=sys.stderr,
            )
            return 1
        else:
            print(
                "Warning: Could not obtain MLIR requirements. " "Skipping sync check.",
                file=sys.stderr,
            )
            return 0

    # Check sync
    missing_or_mismatched = []
    for req in mlir_requirements:
        if req not in build_requires:
            missing_or_mismatched.append(req)

    if missing_or_mismatched:
        print(
            f"Error: The following requirements from {requirements_path}"
            " (or GitHub equivalent) are missing or mismatched in"
            f" {pyproject_path} [build-system].requires:",
            file=sys.stderr,
        )
        for req in missing_or_mismatched:
            print(f"  - {req}", file=sys.stderr)
        print(
            f"Please update [build-system].requires in {pyproject_path} to match.",
            file=sys.stderr,
        )
        return 1

    print("Requirements are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
