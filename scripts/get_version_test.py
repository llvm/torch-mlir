# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import datetime
import os
import subprocess
import sys
import unittest
from unittest import mock

# Add repo root and scripts dir to sys.path so get_version can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import get_version


class TestGetVersion(unittest.TestCase):

    def test_get_package_name(self):
        package_name = get_version.get_package_name()
        self.assertEqual(package_name, "torch-mlir")

    def test_validate_and_parse_tag_valid_stable(self):
        ver, is_dev = get_version.validate_and_parse_tag("v20260831")
        self.assertEqual(ver, "20260831")
        self.assertFalse(is_dev)

        ver, is_dev = get_version.validate_and_parse_tag("refs/tags/v20260831")
        self.assertEqual(ver, "20260831")
        self.assertFalse(is_dev)

    def test_validate_and_parse_tag_valid_dev(self):
        ver, is_dev = get_version.validate_and_parse_tag("v20260831.dev0")
        self.assertEqual(ver, "20260831.dev0")
        self.assertTrue(is_dev)

        ver, is_dev = get_version.validate_and_parse_tag("refs/tags/v20260831.dev12")
        self.assertEqual(ver, "20260831.dev12")
        self.assertTrue(is_dev)

    def test_validate_and_parse_tag_valid_leap_year(self):
        ver, is_dev = get_version.validate_and_parse_tag("v20240229")
        self.assertEqual(ver, "20240229")
        self.assertFalse(is_dev)

    def test_validate_and_parse_tag_invalid_tags(self):
        invalid_tags = [
            "V20260831",  # capital V
            "vv20260831",  # duplicate v
            "vvv20260831",
            "v2026.08.31",  # dotted YYYY.MM.DD
            "v20260831.DEV1",  # capital DEV
            "refs/heads/v20260831",  # branch ref
            "refs/tags/V20260831",  # capital V in refs/tags
            "refs/tags/vv20260831",
            "v20260831-dev",  # hyphen instead of dot
            "v2026083",  # 7 digits
            "v202608311",  # 9 digits
            "v20260231",  # invalid day in February
            "v20260229",  # non-leap year Feb 29
            "v99999999",  # invalid month/day
            "v20261301",  # invalid month 13
            "main",  # branch name
            "release/v20260831",
        ]
        for tag in invalid_tags:
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError) as ctx:
                    get_version.validate_and_parse_tag(tag)
                self.assertIn("Invalid release tag", str(ctx.exception))

    @mock.patch("requests.get")
    def test_get_github_dev_versions(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "assets": [
                {"name": "torch_mlir-20260831.dev0-cp311-cp311-linux_x86_64.whl"},
                {"name": "torch_mlir-20260831.dev1-cp311-cp311-linux_x86_64.whl"},
                {"name": "other_pkg-20260831.dev5-cp311-cp311-linux_x86_64.whl"},
                {"name": "torch-mlir-opt-linux-x86_64"},
            ]
        }
        mock_get.return_value = mock_response

        versions = get_version.get_github_dev_versions("llvm/torch-mlir", "torch-mlir")
        self.assertEqual(versions, ["20260831.dev0", "20260831.dev1"])

    @mock.patch("requests.get")
    def test_get_github_dev_versions_404(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        versions = get_version.get_github_dev_versions("llvm/torch-mlir", "torch-mlir")
        self.assertEqual(versions, [])

    @mock.patch("requests.get")
    def test_get_github_dev_versions_fails_closed_on_http_error(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            get_version.get_github_dev_versions("llvm/torch-mlir", "torch-mlir")
        self.assertIn("HTTP 500", str(ctx.exception))

    @mock.patch("requests.get")
    def test_get_github_dev_versions_fails_closed_on_exception(self, mock_get):
        mock_get.side_effect = Exception("network timeout")
        with self.assertRaises(RuntimeError) as ctx:
            get_version.get_github_dev_versions("llvm/torch-mlir", "torch-mlir")
        self.assertIn("network timeout", str(ctx.exception))

    @mock.patch.dict(os.environ, {"GITHUB_TOKEN": "secret-token"})
    @mock.patch("requests.get")
    def test_get_github_dev_versions_auth_header(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"assets": []}
        mock_get.return_value = mock_response

        get_version.get_github_dev_versions("llvm/torch-mlir", "torch-mlir")
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/llvm/torch-mlir/releases/tags/dev-wheels",
            headers={"Authorization": "token secret-token"},
            timeout=10,
        )

    @mock.patch("requests.get")
    def test_get_pypi_versions(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "releases": {
                "20221212.685": [],
                "20221213.686": [],
            }
        }
        mock_get.return_value = mock_response

        versions = get_version.get_pypi_versions("torch-mlir")
        self.assertIn("20221213.686", versions)
        self.assertIn("20221212.685", versions)

    @mock.patch("requests.get")
    def test_get_pypi_versions_404(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        versions = get_version.get_pypi_versions("torch-mlir")
        self.assertEqual(versions, [])

    @mock.patch("requests.get")
    def test_get_pypi_versions_fails_closed_on_network_error(self, mock_get):
        mock_get.side_effect = Exception("PyPI unreachable")
        with self.assertRaises(RuntimeError) as ctx:
            get_version.get_pypi_versions("torch-mlir")
        self.assertIn("PyPI unreachable", str(ctx.exception))

    @mock.patch("requests.get")
    def test_get_pypi_versions_fails_closed_on_http_error(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            get_version.get_pypi_versions("torch-mlir")
        self.assertIn("HTTP 503", str(ctx.exception))

    @mock.patch("requests.get")
    def test_get_pypi_versions_fails_closed_on_json_error(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            get_version.get_pypi_versions("torch-mlir")
        self.assertIn("Failed to decode JSON", str(ctx.exception))

    @mock.patch("scripts.get_version.get_pypi_versions")
    def test_verify_latest_version_success(self, mock_pypi):
        mock_pypi.return_value = ["20221212.685", "20221213.686"]
        # Newer date-based version should pass
        get_version.verify_latest_version("20260831", "torch-mlir")
        get_version.verify_latest_version("20260831.dev0", "torch-mlir")

    @mock.patch("scripts.get_version.get_pypi_versions")
    def test_verify_latest_version_failure_older(self, mock_pypi):
        mock_pypi.return_value = ["20221212.685", "20221213.686"]
        # Older version should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            get_version.verify_latest_version("20220101", "torch-mlir")
        self.assertIn(
            "not greater than the latest existing PyPI release", str(ctx.exception)
        )

    @mock.patch("scripts.get_version.get_pypi_versions")
    def test_verify_latest_version_failure_old_date_format(self, mock_pypi):
        mock_pypi.return_value = ["20221213.686"]
        # YYYY.MM.DD evaluates as 2026.x which is less than 20221213.x
        with self.assertRaises(ValueError) as ctx:
            get_version.verify_latest_version("2026.08.31", "torch-mlir")
        self.assertIn(
            "not greater than the latest existing PyPI release", str(ctx.exception)
        )

    @mock.patch("scripts.get_version.get_pypi_versions")
    def test_verify_latest_version_equal_allowed_for_reruns(self, mock_pypi):
        mock_pypi.return_value = ["20260831"]
        # Equal version should pass to support idempotent retries / missing-file uploads
        get_version.verify_latest_version("20260831", "torch-mlir")

    @mock.patch("scripts.get_version.get_pypi_versions")
    def test_verify_latest_version_no_existing(self, mock_pypi):
        mock_pypi.return_value = []
        # No existing releases -> should not raise
        get_version.verify_latest_version("20260831", "torch-mlir")

    @mock.patch("scripts.get_version.get_github_dev_versions")
    def test_get_next_dev_version_first(self, mock_dev_versions):
        mock_dev_versions.return_value = []
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        version = get_version.get_next_dev_version("torch-mlir", "llvm/torch-mlir")
        self.assertEqual(version, f"{today}.dev0")

    @mock.patch("scripts.get_version.get_github_dev_versions")
    def test_get_next_dev_version_increment(self, mock_dev_versions):
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        mock_dev_versions.return_value = [
            f"{today}.dev0",
            f"{today}.dev1",
            f"{today}.dev2",
        ]
        version = get_version.get_next_dev_version("torch-mlir", "llvm/torch-mlir")
        self.assertEqual(version, f"{today}.dev3")

    @mock.patch("scripts.get_version.get_github_dev_versions")
    def test_get_next_dev_version_canonicalizes_versions(self, mock_dev_versions):
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        # Test versions with different representation (e.g. .dev05), non-dev release, and invalid string
        mock_dev_versions.return_value = [
            f"{today}.dev0",
            f"{today}.dev05",
            f"{today}",  # non-dev release
            "invalid-version-string",
        ]
        version = get_version.get_next_dev_version("torch-mlir", "llvm/torch-mlir")
        # max dev is 5, so next dev is 6
        self.assertEqual(version, f"{today}.dev6")

    @mock.patch("scripts.get_version.get_github_dev_versions")
    def test_get_next_dev_version_ignores_other_dates(self, mock_dev_versions):
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        mock_dev_versions.return_value = ["20200101.dev5", "20200101.dev6"]
        version = get_version.get_next_dev_version("torch-mlir", "llvm/torch-mlir")
        self.assertEqual(version, f"{today}.dev0")

    @mock.patch("subprocess.run")
    def test_resolve_and_verify_tag_success(self, mock_run):
        expected_sha = "a" * 40
        mock_run.side_effect = [
            mock.Mock(stdout=f"{expected_sha}\n", returncode=0),
            mock.Mock(stdout="", returncode=0),
        ]
        sha = get_version.resolve_and_verify_tag("v20260831")
        self.assertEqual(sha, expected_sha)
        self.assertEqual(mock_run.call_count, 2)
        mock_run.assert_any_call(
            ["git", "rev-parse", "--verify", "refs/tags/v20260831^{commit}"],
            capture_output=True,
            text=True,
            check=True,
        )
        mock_run.assert_any_call(
            ["git", "merge-base", "--is-ancestor", expected_sha, "origin/main"],
            capture_output=True,
            text=True,
            check=True,
        )

    @mock.patch("subprocess.run")
    def test_resolve_and_verify_tag_not_found(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "rev-parse"], stderr="fatal: Needed a single revision"
        )
        with self.assertRaises(ValueError) as ctx:
            get_version.resolve_and_verify_tag("v20260831")
        self.assertIn("Failed to resolve tag", str(ctx.exception))

    @mock.patch("subprocess.run")
    def test_resolve_and_verify_tag_not_ancestor(self, mock_run):
        expected_sha = "b" * 40
        mock_run.side_effect = [
            mock.Mock(stdout=f"{expected_sha}\n", returncode=0),
            subprocess.CalledProcessError(1, ["git", "merge-base"], stderr=""),
        ]
        with self.assertRaises(ValueError) as ctx:
            get_version.resolve_and_verify_tag("v20260831")
        self.assertIn("not an ancestor", str(ctx.exception))

    def test_calculate_version_pull_request(self):
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="pull_request",
            ref="refs/pull/123/merge",
            tag=None,
            package="torch-mlir",
        )
        self.assertEqual(v, "0.0.0")
        self.assertEqual(ref, "refs/pull/123/merge")
        self.assertEqual(pub_gh, "false")
        self.assertEqual(pub_pypi, "false")

    @mock.patch("scripts.get_version.resolve_and_verify_tag")
    def test_calculate_version_workflow_dispatch_tag(self, mock_resolve_tag):
        expected_sha = "a" * 40
        mock_resolve_tag.return_value = expected_sha
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="workflow_dispatch",
            ref="refs/heads/main",
            tag="v20260831",
            package="torch-mlir",
        )
        self.assertEqual(v, "20260831")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "true")
        mock_resolve_tag.assert_called_once_with("v20260831")

    @mock.patch("scripts.get_version.resolve_and_verify_tag")
    def test_calculate_version_workflow_dispatch_dev_tag(self, mock_resolve_tag):
        expected_sha = "b" * 40
        mock_resolve_tag.return_value = expected_sha
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="workflow_dispatch",
            ref="refs/heads/main",
            tag="v20260831.dev1",
            package="torch-mlir",
        )
        self.assertEqual(v, "20260831.dev1")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "false")
        mock_resolve_tag.assert_called_once_with("v20260831.dev1")

    def test_calculate_version_workflow_dispatch_invalid_tag(self):
        with self.assertRaises(ValueError):
            get_version.calculate_version(
                event="workflow_dispatch",
                ref="refs/heads/main",
                tag="V20260831",
                package="torch-mlir",
            )

    @mock.patch("scripts.get_version.resolve_ref_sha")
    @mock.patch("scripts.get_version.get_next_dev_version")
    def test_calculate_version_workflow_dispatch_main_dev(
        self, mock_dev, mock_resolve_ref
    ):
        expected_sha = "c" * 40
        mock_dev.return_value = "20260831.dev0"
        mock_resolve_ref.return_value = expected_sha
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="workflow_dispatch",
            ref="refs/heads/main",
            tag=None,
            package="torch-mlir",
        )
        self.assertEqual(v, "20260831.dev0")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "false")
        mock_resolve_ref.assert_called_once_with("origin/main")

    def test_calculate_version_workflow_dispatch_feature_branch(self):
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="workflow_dispatch",
            ref="refs/heads/feature-branch",
            tag=None,
            package="torch-mlir",
        )
        self.assertEqual(v, "0.0.0")
        self.assertEqual(ref, "refs/heads/feature-branch")
        self.assertEqual(pub_gh, "false")
        self.assertEqual(pub_pypi, "false")

    @mock.patch("scripts.get_version.resolve_ref_sha")
    @mock.patch("datetime.datetime")
    def test_calculate_version_schedule_first_of_month(
        self, mock_datetime, mock_resolve_ref
    ):
        expected_sha = "d" * 40
        mock_resolve_ref.return_value = expected_sha
        mock_now = mock.Mock()
        mock_now.day = 1
        mock_now.strftime.return_value = "20260901"
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = datetime.timezone

        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="schedule",
            ref="refs/heads/main",
            tag=None,
            package="torch-mlir",
        )
        self.assertEqual(v, "20260901")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "true")
        mock_resolve_ref.assert_called_once_with("origin/main")

    @mock.patch("scripts.get_version.resolve_ref_sha")
    @mock.patch("datetime.datetime")
    @mock.patch("scripts.get_version.get_next_dev_version")
    def test_calculate_version_schedule_mid_month(
        self, mock_dev, mock_datetime, mock_resolve_ref
    ):
        expected_sha = "e" * 40
        mock_resolve_ref.return_value = expected_sha
        mock_now = mock.Mock()
        mock_now.day = 15
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = datetime.timezone
        mock_dev.return_value = "20260915.dev0"

        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="schedule",
            ref="refs/heads/main",
            tag=None,
            package="torch-mlir",
        )
        self.assertEqual(v, "20260915.dev0")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "false")
        mock_resolve_ref.assert_called_once_with("origin/main")

    @mock.patch("scripts.get_version.resolve_and_verify_tag")
    def test_calculate_version_schedule_tag(self, mock_resolve_tag):
        expected_sha = "f" * 40
        mock_resolve_tag.return_value = expected_sha
        v, ref, pub_gh, pub_pypi = get_version.calculate_version(
            event="schedule",
            ref="refs/heads/main",
            tag="v20260901",
            package="torch-mlir",
        )
        self.assertEqual(v, "20260901")
        self.assertEqual(ref, expected_sha)
        self.assertEqual(pub_gh, "true")
        self.assertEqual(pub_pypi, "true")
        mock_resolve_tag.assert_called_once_with("v20260901")


class TestMainCLI(unittest.TestCase):

    @mock.patch("sys.argv", ["get_version.py", "--event", "pull_request"])
    def test_main_pull_request(self):
        with mock.patch("builtins.print") as mock_print:
            get_version.main()
            mock_print.assert_any_call("version=0.0.0")
            mock_print.assert_any_call("ref=refs/heads/main")
            mock_print.assert_any_call("should_publish_gh=false")
            mock_print.assert_any_call("should_publish_pypi=false")

    @mock.patch("scripts.get_version.resolve_and_verify_tag")
    @mock.patch("scripts.get_version.verify_latest_version")
    @mock.patch(
        "sys.argv",
        ["get_version.py", "--event", "workflow_dispatch", "--tag", "v20260831"],
    )
    def test_main_workflow_dispatch(self, mock_verify, mock_resolve_tag):
        expected_sha = "a" * 40
        mock_resolve_tag.return_value = expected_sha
        with mock.patch("builtins.print") as mock_print:
            get_version.main()
            mock_verify.assert_called_once_with("20260831", "torch-mlir")
            mock_print.assert_any_call("version=20260831")
            mock_print.assert_any_call(f"ref={expected_sha}")
            mock_print.assert_any_call("should_publish_gh=true")
            mock_print.assert_any_call("should_publish_pypi=true")

    @mock.patch("scripts.get_version.resolve_and_verify_tag")
    @mock.patch("scripts.get_version.verify_latest_version")
    @mock.patch(
        "sys.argv",
        [
            "get_version.py",
            "--event",
            "workflow_dispatch",
            "--tag",
            "v20260831",
            "--gha",
        ],
    )
    def test_main_gha_output(self, mock_verify, mock_resolve_tag, tmp_path=None):
        import tempfile

        expected_sha = "a" * 40
        mock_resolve_tag.return_value = expected_sha
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
            output_file = tf.name

        try:
            with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                get_version.main()

            with open(output_file) as f:
                content = f.read()

            self.assertIn("version=20260831\n", content)
            self.assertIn(f"ref={expected_sha}\n", content)
            self.assertIn("should_publish_gh=true\n", content)
            self.assertIn("should_publish_pypi=true\n", content)
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
