# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import hashlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

# Add repo root to sys.path so scripts module can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import prepare_pypi_publish


class TestPreparePyPIPublish(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dist_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_wheel(self, filename: str, content: bytes) -> tuple[pathlib.Path, str]:
        path = self.dist_dir / filename
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest().lower()
        return path, sha256

    def test_compute_sha256(self):
        wheel_path, expected_sha = self._create_wheel(
            "test.whl", b"wheel content for sha256"
        )
        self.assertEqual(prepare_pypi_publish.compute_sha256(wheel_path), expected_sha)

    @mock.patch("requests.get")
    def test_fetch_pypi_release_files_success(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "releases": {
                "20260901": [
                    {
                        "filename": "pkg-20260901-cp311-manylinux.whl",
                        "digests": {"sha256": "abcdef123456"},
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        files = prepare_pypi_publish.fetch_pypi_release_files("torch-mlir", "20260901")
        self.assertEqual(files, {"pkg-20260901-cp311-manylinux.whl": "abcdef123456"})

    @mock.patch("requests.get")
    def test_fetch_pypi_release_files_404(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        files = prepare_pypi_publish.fetch_pypi_release_files("torch-mlir", "20260901")
        self.assertEqual(files, {})

    @mock.patch("requests.get")
    def test_fetch_pypi_release_files_error(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            prepare_pypi_publish.fetch_pypi_release_files("torch-mlir", "20260901")
        self.assertIn("HTTP 500", str(ctx.exception))

    @mock.patch("requests.get")
    def test_fetch_pypi_release_files_network_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            prepare_pypi_publish.fetch_pypi_release_files("torch-mlir", "20260901")
        self.assertIn("Connection refused", str(ctx.exception))

    @mock.patch("scripts.prepare_pypi_publish.fetch_pypi_release_files")
    def test_prepare_publish_fresh_release(self, mock_fetch):
        mock_fetch.return_value = {}

        w1, _ = self._create_wheel(
            "torch_mlir-20260901-cp311-linux.whl", b"linux wheel"
        )
        w2, _ = self._create_wheel(
            "torch_mlir-20260901-cp311-macos.whl", b"macos wheel"
        )

        should_upload, state = prepare_pypi_publish.prepare_publish(
            self.dist_dir, "torch-mlir", "20260901"
        )
        self.assertTrue(should_upload)
        self.assertEqual(state, "FRESH_RELEASE")
        self.assertTrue(w1.exists())
        self.assertTrue(w2.exists())

    @mock.patch("scripts.prepare_pypi_publish.fetch_pypi_release_files")
    def test_prepare_publish_already_completed(self, mock_fetch):
        w1, sha1 = self._create_wheel(
            "torch_mlir-20260901-cp311-linux.whl", b"linux wheel"
        )
        w2, sha2 = self._create_wheel(
            "torch_mlir-20260901-cp311-macos.whl", b"macos wheel"
        )

        mock_fetch.return_value = {
            "torch_mlir-20260901-cp311-linux.whl": sha1,
            "torch_mlir-20260901-cp311-macos.whl": sha2,
        }

        should_upload, state = prepare_pypi_publish.prepare_publish(
            self.dist_dir, "torch-mlir", "20260901"
        )
        self.assertFalse(should_upload)
        self.assertEqual(state, "ALREADY_COMPLETED")
        self.assertFalse(w1.exists())
        self.assertFalse(w2.exists())

    @mock.patch("scripts.prepare_pypi_publish.fetch_pypi_release_files")
    def test_prepare_publish_incomplete_retry(self, mock_fetch):
        w1, sha1 = self._create_wheel(
            "torch_mlir-20260901-cp311-linux.whl", b"linux wheel"
        )
        w2, _ = self._create_wheel(
            "torch_mlir-20260901-cp311-macos.whl", b"macos wheel"
        )

        # w1 already uploaded, w2 missing
        mock_fetch.return_value = {
            "torch_mlir-20260901-cp311-linux.whl": sha1,
        }

        should_upload, state = prepare_pypi_publish.prepare_publish(
            self.dist_dir, "torch-mlir", "20260901"
        )
        self.assertTrue(should_upload)
        self.assertEqual(state, "INCOMPLETE_RETRY")
        self.assertFalse(w1.exists())  # existing wheel pruned
        self.assertTrue(w2.exists())  # missing wheel kept

    @mock.patch("scripts.prepare_pypi_publish.fetch_pypi_release_files")
    def test_prepare_publish_content_collision_fails(self, mock_fetch):
        self._create_wheel("torch_mlir-20260901-cp311-linux.whl", b"new modified wheel")

        # PyPI has different hash for same filename
        mock_fetch.return_value = {
            "torch_mlir-20260901-cp311-linux.whl": "different_hash_on_pypi",
        }

        with self.assertRaises(ValueError) as ctx:
            prepare_pypi_publish.prepare_publish(
                self.dist_dir, "torch-mlir", "20260901"
            )
        self.assertIn("Content collision", str(ctx.exception))
        self.assertIn("PyPI artifacts are immutable", str(ctx.exception))

    def test_prepare_publish_no_wheels(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            prepare_pypi_publish.prepare_publish(
                self.dist_dir, "torch-mlir", "20260901"
            )
        self.assertIn("No wheels found", str(ctx.exception))


class TestPreparePyPIPublishMainCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dist_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("scripts.prepare_pypi_publish.fetch_pypi_release_files")
    def test_main_cli_gha(self, mock_fetch):
        mock_fetch.return_value = {}
        (self.dist_dir / "torch_mlir-20260901-cp311-linux.whl").write_bytes(b"content")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
            output_file = tf.name

        try:
            with mock.patch(
                "sys.argv",
                [
                    "prepare_pypi_publish.py",
                    "--dist-dir",
                    str(self.dist_dir),
                    "--version",
                    "20260901",
                    "--gha",
                ],
            ):
                with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                    prepare_pypi_publish.main()

            with open(output_file) as f:
                content = f.read()

            self.assertIn("should_upload=true\n", content)
            self.assertIn("release_state=FRESH_RELEASE\n", content)
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
