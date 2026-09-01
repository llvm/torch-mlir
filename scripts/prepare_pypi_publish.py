"""Preflight check and preparation for PyPI wheel publishing.

Enforces PyPI release immutability and supports idempotent missing-file retries:
1. Queries PyPI for existing files and SHA256 digests under the target version.
2. Compares local wheel files against PyPI:
   - If a file exists with matching SHA256: removes it from the local dist/ directory
     so it is not re-uploaded.
   - If a file exists with a DIFFERENT SHA256: fails closed with ValueError
     (content collision; PyPI artifacts are immutable).
   - If a file does not exist: leaves it in dist/ to be uploaded.
3. If all files already exist on PyPI, outputs should_upload=false to skip publishing.
"""

import argparse
import hashlib
import os
import pathlib
import sys

import requests

CHUNK_SIZE = 65536


def compute_sha256(filepath: pathlib.Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_pypi_release_files(package_name: str, version_str: str) -> dict[str, str]:
    """Fetch existing release files and SHA256 digests for a version from PyPI.

    Returns a dict mapping filename -> sha256.
    Fails closed on network or server errors.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=15)
    except Exception as e:
        raise RuntimeError(
            f"Failed to query PyPI metadata for '{package_name}': {e}"
        ) from e

    if response.status_code == 404:
        # Package not found or has no releases yet
        return {}

    if response.status_code != 200:
        raise RuntimeError(
            f"PyPI returned HTTP {response.status_code} when querying metadata for '{package_name}': {response.text}"
        )

    data = response.json()
    releases = data.get("releases", {})
    files_info = releases.get(version_str, [])

    file_hashes = {}
    for item in files_info:
        filename = item.get("filename")
        sha256 = item.get("digests", {}).get("sha256")
        if filename and sha256:
            file_hashes[filename] = sha256.lower()

    return file_hashes


def prepare_publish(
    dist_dir: pathlib.Path, package_name: str, version_str: str
) -> tuple[bool, str]:
    """Inspect local wheels, compare against PyPI, and prepare dist_dir for upload.

    Returns (should_upload, release_state).
    """
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"Distribution directory '{dist_dir}' not found.")

    local_wheels = sorted(dist_dir.glob("*.whl"))
    if not local_wheels:
        raise FileNotFoundError(
            f"No wheels found in distribution directory '{dist_dir}'."
        )

    pypi_files = fetch_pypi_release_files(package_name, version_str)

    already_published = []
    to_upload = []

    for wheel_path in local_wheels:
        filename = wheel_path.name
        local_sha = compute_sha256(wheel_path).lower()

        if filename in pypi_files:
            pypi_sha = pypi_files[filename]
            if local_sha != pypi_sha:
                raise ValueError(
                    f"Content collision for '{filename}': local SHA256 ({local_sha}) "
                    f"does not match PyPI SHA256 ({pypi_sha}). "
                    f"PyPI artifacts are immutable; bump the version to release modified content."
                )
            # Hashes match: file is already published on PyPI.
            # Remove it locally so gh-action-pypi-publish skips it.
            wheel_path.unlink()
            already_published.append(filename)
        else:
            to_upload.append(filename)

    total_count = len(local_wheels)
    upload_count = len(to_upload)

    if upload_count == 0:
        state = "ALREADY_COMPLETED"
        print(
            f"PyPI Preflight: All {total_count} wheel(s) for version '{version_str}' "
            f"are already published on PyPI with matching SHA256 hashes. "
            f"Skipping PyPI upload."
        )
        return False, state

    if len(already_published) > 0:
        state = "INCOMPLETE_RETRY"
        print(
            f"PyPI Preflight: {len(already_published)} of {total_count} wheel(s) for version '{version_str}' "
            f"already exist on PyPI with matching SHA256 hashes and were pruned. "
            f"Uploading {upload_count} missing wheel(s): {to_upload}"
        )
        return True, state

    state = "FRESH_RELEASE"
    print(
        f"PyPI Preflight: Fresh release. Uploading {upload_count} wheel(s) for version '{version_str}' to PyPI: {to_upload}"
    )
    return True, state


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PyPI publication and enforce preflight contract."
    )
    parser.add_argument(
        "--dist-dir", default="dist", help="Directory containing wheel artifacts"
    )
    parser.add_argument("--package", default="torch-mlir", help="PyPI package name")
    parser.add_argument("--version", required=True, help="Target package version")
    parser.add_argument(
        "--gha", action="store_true", help="Output variables for GitHub Actions"
    )

    args = parser.parse_args()

    dist_dir = pathlib.Path(args.dist_dir).resolve()
    should_upload, state = prepare_publish(dist_dir, args.package, args.version)

    should_upload_str = "true" if should_upload else "false"

    if args.gha:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"should_upload={should_upload_str}\n")
                f.write(f"release_state={state}\n")

    print(f"should_upload={should_upload_str}")
    print(f"release_state={state}")


if __name__ == "__main__":
    main()
