"""Script to calculate the version for the torch-mlir Python package.

Adapted from the HEIR release tooling. Produces a date-based version
(``YYYY.MM.DD``) for tagged/scheduled releases and an auto-incrementing
``YYYY.MM.DD.devN`` for development releases off ``main``. The version is fed to
``setup.py`` via the ``TORCH_MLIR_PYTHON_PACKAGE_VERSION`` environment variable.
"""

import argparse
import datetime
import os
import pathlib
import re
import sys
import tomllib

import requests

# Branch that development (.devN) releases are cut from.
DEV_BRANCH_REF = "refs/heads/main"

# Fallback package name if pyproject.toml has no [project] name.
DEFAULT_PACKAGE = "torch-mlir"


def get_package_name():
    """Read the package name from pyproject.toml, falling back to the default."""
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["name"]
    except (KeyError, FileNotFoundError):
        return DEFAULT_PACKAGE


def get_github_dev_versions(repo, package_name):
    """Fetch versions of dev wheels from GitHub dev-wheels release."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/dev-wheels"
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            versions = []
            # PEP 427: distribution name is escaped by replacing runs of
            # non-alphanumeric characters with a single underscore.
            escaped_package = re.sub(r"[^a-zA-Z0-9]+", "_", package_name).lower()

            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if not name.endswith(".whl"):
                    continue
                # Split by '-' to get distribution name and version
                parts = name.split("-")
                if len(parts) >= 2:
                    dist_name = re.sub(r"[^a-zA-Z0-9]+", "_", parts[0]).lower()
                    if dist_name == escaped_package:
                        version = parts[1]
                        versions.append(version)
            return versions
    except Exception as e:
        print(f"Error fetching from GitHub: {e}", file=sys.stderr)
    return []


def get_next_dev_version(package_name, repo=None):
    """Calculate the next .devN version for today's date."""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y.%m.%d")
    versions = get_github_dev_versions(repo, package_name) if repo else []

    # Find versions matching today's date and the .dev suffix
    pattern = re.compile(rf"^{re.escape(today)}\.dev(\d+)$")
    max_dev = -1

    for v in versions:
        match = pattern.match(v)
        if match:
            max_dev = max(max_dev, int(match.group(1)))

    next_dev = max_dev + 1
    return f"{today}.dev{next_dev}"


def calculate_version(event, ref, tag, package, repo=None):
    version = "0.0.0"
    should_publish_gh = "false"
    should_publish_pypi = "false"

    # NOTE: if/elif rather than match/case so the repo's pinned (py38-target)
    # black can parse this file.
    if event == "workflow_dispatch":
        if tag:
            # Manual release of an existing tag; use for example when the release
            # workflow fails to trigger the wheel upload.
            version = tag.lstrip("v")
            should_publish_gh = "true"
            if "dev" not in version:
                should_publish_pypi = "true"
        elif ref == DEV_BRANCH_REF:
            # For dev releases
            version = get_next_dev_version(package, repo)
            should_publish_gh = "true"

    elif event == "schedule":
        should_publish_gh = "true"
        if tag:
            version = tag.lstrip("v")
            if "dev" not in version:
                should_publish_pypi = "true"
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.day == 1:
                version = now.strftime("%Y.%m.%d")
                should_publish_pypi = "true"
            else:
                version = get_next_dev_version(package, repo)

    elif event == "pull_request":
        pass  # use defaults which are all safe

    return version, should_publish_gh, should_publish_pypi


def main():
    parser = argparse.ArgumentParser(
        description="Calculate torch-mlir package version."
    )
    parser.add_argument(
        "--event",
        default="workflow_dispatch",
        help="GitHub event name (e.g., schedule, workflow_dispatch)",
    )
    parser.add_argument(
        "--ref",
        default="refs/heads/main",
        help="GitHub ref (e.g., refs/heads/main)",
    )
    parser.add_argument("--tag", help="Release tag name")
    parser.add_argument(
        "--package",
        default=None,
        help="PyPI package name (defaults to the name in pyproject.toml)",
    )
    parser.add_argument("--gha", action="store_true", help="Output for GitHub Actions")

    args = parser.parse_args()

    package = args.package or get_package_name()
    repo = os.environ.get("GITHUB_REPOSITORY")
    version, should_publish_gh, should_publish_pypi = calculate_version(
        args.event, args.ref, args.tag, package, repo
    )

    if args.gha:
        # Writing to GITHUB_OUTPUT if available
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"version={version}\n")
                f.write(f"should_publish_gh={should_publish_gh}\n")
                f.write(f"should_publish_pypi={should_publish_pypi}\n")

    print(f"version={version}")
    print(f"should_publish_gh={should_publish_gh}")
    print(f"should_publish_pypi={should_publish_pypi}")


if __name__ == "__main__":
    main()
