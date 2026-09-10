"""Script to calculate the version for the torch-mlir Python package.

Produces a date-based version (``YYYYMMDD``) for tagged/scheduled releases and
an auto-incrementing ``YYYYMMDD.devN`` for development releases off ``main``.
The version is fed to ``setup.py`` via the ``TORCH_MLIR_PYTHON_PACKAGE_VERSION``
environment variable.
"""

import argparse
import datetime
import os
import pathlib
import re
import sys
import tomllib

import packaging.version
import requests

# Branch that development (.devN) releases are cut from.
DEV_BRANCH_REF = "refs/heads/main"

# Fallback package name if pyproject.toml has no [project] name.
DEFAULT_PACKAGE = "torch-mlir"

# Strict regex for case-sensitive vYYYYMMDD or vYYYYMMDD.devN tag names.
TAG_REGEX = re.compile(r"^(?:refs/tags/)?(v\d{8}(?:\.dev\d+)?)$")


def get_package_name():
    """Read the package name from pyproject.toml, falling back to the default."""
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["name"]
    except (KeyError, FileNotFoundError):
        return DEFAULT_PACKAGE


def validate_and_parse_tag(tag: str) -> tuple[str, bool]:
    """Validate a case-sensitive refs/tags/vYYYYMMDD or vYYYYMMDD(.devN) tag.

    Rejects branch/tag ambiguity, duplicate leading 'v's, wrong capitalization,
    and invalid formats. Parses with packaging.version.Version and returns
    (version_str, is_devrelease).
    """
    match = TAG_REGEX.match(tag)
    if not match:
        raise ValueError(
            f"Invalid release tag '{tag}'. Expected exact format 'vYYYYMMDD' or "
            f"'vYYYYMMDD.devN' (e.g. 'v20260831')."
        )
    tag_clean = match.group(1)
    version_str = tag_clean.removeprefix("v")
    parsed = packaging.version.Version(version_str)
    return str(parsed), parsed.is_devrelease


def get_github_dev_versions(repo, package_name):
    """Fetch versions of dev wheels from GitHub dev-wheels release.

    Fails closed (raises RuntimeError) if the GitHub API query fails with any status
    code other than 404 (release tag does not exist yet) or if a network error occurs.
    """
    url = f"https://api.github.com/repos/{repo}/releases/tags/dev-wheels"
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        raise RuntimeError(
            f"Failed to query GitHub dev-wheels release for '{repo}': {e}"
        ) from e

    if response.status_code == 404:
        return []
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned HTTP {response.status_code} when fetching dev releases "
            f"for '{repo}': {response.text}"
        )

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


def get_pypi_versions(package_name):
    """Fetch all release versions for a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return list(data.get("releases", {}).keys())
    except Exception as e:
        print(f"Error fetching from PyPI: {e}", file=sys.stderr)
    return []


def verify_latest_version(version_str, package_name):
    """Verify that the chosen release version is not older than existing PyPI releases.

    NOTE: This check enforces that new releases are greater than or equal to all
    previously published PyPI versions under PEP 440 ordering. This assumes a
    date-based versioning scheme (e.g. YYYYMMDD). This would need to change if
    torch-mlir switches to stable releases with patches (e.g., releasing a patch
    fix 1.0.1 after 1.1.0 has already been published).
    """
    pypi_versions = get_pypi_versions(package_name)
    if not pypi_versions:
        return

    parsed_target = packaging.version.parse(version_str)
    parsed_existing = [packaging.version.parse(v) for v in pypi_versions]
    latest_existing = max(parsed_existing)

    if parsed_target < latest_existing:
        raise ValueError(
            f"Calculated version '{version_str}' is not greater than the latest "
            f"existing PyPI release '{latest_existing}'. "
            f"New releases must not be older than existing PyPI releases."
        )


def get_next_dev_version(package_name, repo=None):
    """Calculate the next .devN version for today's date (YYYYMMDD.devN)."""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    target_base = packaging.version.parse(today).base_version
    versions = get_github_dev_versions(repo, package_name) if repo else []

    max_dev = -1
    for v in versions:
        try:
            parsed = packaging.version.parse(v)
            if (
                parsed.is_devrelease
                and parsed.base_version == target_base
                and parsed.dev is not None
            ):
                max_dev = max(max_dev, parsed.dev)
        except packaging.version.InvalidVersion:
            continue

    next_dev = max_dev + 1
    return f"{today}.dev{next_dev}"


def calculate_version(event, ref, tag, package, repo=None):
    version = "0.0.0"
    target_ref = DEV_BRANCH_REF
    should_publish_gh = "false"
    should_publish_pypi = "false"

    # NOTE: if/elif rather than match/case so the repo's pinned (py38-target)
    # black can parse this file.
    if event == "workflow_dispatch":
        if tag:
            # Manual release of an existing tag; validate and parse with Version
            version, is_dev = validate_and_parse_tag(tag)
            clean_tag = tag.removeprefix("refs/tags/")
            target_ref = f"refs/tags/{clean_tag}"
            should_publish_gh = "true"
            if not is_dev:
                should_publish_pypi = "true"
        elif ref == DEV_BRANCH_REF:
            # For dev releases off main
            version = get_next_dev_version(package, repo)
            target_ref = DEV_BRANCH_REF
            should_publish_gh = "true"
        else:
            target_ref = ref

    elif event == "schedule":
        should_publish_gh = "true"
        target_ref = DEV_BRANCH_REF
        if tag:
            version, is_dev = validate_and_parse_tag(tag)
            clean_tag = tag.removeprefix("refs/tags/")
            target_ref = f"refs/tags/{clean_tag}"
            if not is_dev:
                should_publish_pypi = "true"
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.day == 1:
                version = now.strftime("%Y%m%d")
                should_publish_pypi = "true"
            else:
                version = get_next_dev_version(package, repo)

    elif event == "pull_request":
        target_ref = ref  # use PR ref

    return version, target_ref, should_publish_gh, should_publish_pypi


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
    version, target_ref, should_publish_gh, should_publish_pypi = calculate_version(
        args.event, args.ref, args.tag, package, repo
    )

    if should_publish_pypi == "true":
        verify_latest_version(version, package)

    if args.gha:
        # Writing to GITHUB_OUTPUT if available
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"version={version}\n")
                f.write(f"ref={target_ref}\n")
                f.write(f"should_publish_gh={should_publish_gh}\n")
                f.write(f"should_publish_pypi={should_publish_pypi}\n")

    print(f"version={version}")
    print(f"ref={target_ref}")
    print(f"should_publish_gh={should_publish_gh}")
    print(f"should_publish_pypi={should_publish_pypi}")


if __name__ == "__main__":
    main()
