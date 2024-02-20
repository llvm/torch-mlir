# Based on code Copyright (c) Advanced Micro Devices, Inc.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s --output %t

from glob import glob
from pathlib import Path

import logging
import sys
import unittest

import onnx

from _torch_mlir_config import (
    configure_context,
    ir,
    onnx_importer,
)

# Accept the output path on the command line or default to a sibling
# to this file. We have to pop this off explicitly or else unittest
# won't understand.
if len(sys.argv) > 1 and sys.argv[1] == "--output":
    OUTPUT_PATH = Path(sys.argv[2])
    del sys.argv[1:3]
else:
    OUTPUT_PATH = Path(__file__).resolve().parent / "output"


# TODO: Add some verification and overrides. For now, just use the
# onnx package install for onnx test files, since they were nice
# enough to include the test suite in the deployable.
import onnx.backend.test.data

ONNX_TEST_DATA_DIR = Path(onnx.backend.test.__file__).resolve().parent / "data"
print(f"ONNX Test Data Dir: {ONNX_TEST_DATA_DIR}")
ONNX_REL_PATHS = glob(f"**/*.onnx", root_dir=ONNX_TEST_DATA_DIR, recursive=True)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TEST_CAST_XFAILS = [
    "light_light_bvlc_alexnet",
    "light_light_inception_v1",
    "light_light_squeezenet",
    "light_light_vgg19",
    "node_test_affine_grid_2d_align_corners_expanded_model",
    "node_test_affine_grid_2d_expanded_model",
    "node_test_affine_grid_3d_align_corners_expanded_model",
    "node_test_affine_grid_3d_expanded_model",
    "node_test_ai_onnx_ml_label_encoder_string_int_model",
    "node_test_ai_onnx_ml_label_encoder_string_int_no_default_model",
    "node_test_ai_onnx_ml_label_encoder_tensor_mapping_model",
    "node_test_ai_onnx_ml_label_encoder_tensor_value_only_mapping_model",
    "node_test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_model",
    "node_test_cast_FLOAT16_to_FLOAT8E4M3FN_model",
    "node_test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_model",
    "node_test_cast_FLOAT16_to_FLOAT8E5M2_model",
    "node_test_cast_FLOAT8E4M3FNUZ_to_FLOAT16_model",
    "node_test_cast_FLOAT8E4M3FNUZ_to_FLOAT_model",
    "node_test_cast_FLOAT8E4M3FN_to_FLOAT16_model",
    "node_test_cast_FLOAT8E4M3FN_to_FLOAT_model",
    "node_test_cast_FLOAT8E5M2FNUZ_to_FLOAT16_model",
    "node_test_cast_FLOAT8E5M2FNUZ_to_FLOAT_model",
    "node_test_cast_FLOAT8E5M2_to_FLOAT16_model",
    "node_test_cast_FLOAT8E5M2_to_FLOAT_model",
    "node_test_cast_FLOAT_to_FLOAT8E4M3FNUZ_model",
    "node_test_cast_FLOAT_to_FLOAT8E4M3FN_model",
    "node_test_cast_FLOAT_to_FLOAT8E5M2FNUZ_model",
    "node_test_cast_FLOAT_to_FLOAT8E5M2_model",
    "node_test_cast_FLOAT_to_STRING_model",
    "node_test_cast_STRING_to_FLOAT_model",
    "node_test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_model",
    "node_test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_model",
    "node_test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_model",
    "node_test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_model",
    "node_test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_model",
    "node_test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_model",
    "node_test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_model",
    "node_test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_model",
    "node_test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded_model",
    "node_test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_model",
    "node_test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded_model",
    "node_test_castlike_FLOAT8E4M3FN_to_FLOAT_model",
    "node_test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded_model",
    "node_test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_model",
    "node_test_castlike_FLOAT8E5M2_to_FLOAT_expanded_model",
    "node_test_castlike_FLOAT8E5M2_to_FLOAT_model",
    "node_test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded_model",
    "node_test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_model",
    "node_test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded_model",
    "node_test_castlike_FLOAT_to_FLOAT8E4M3FN_model",
    "node_test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded_model",
    "node_test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_model",
    "node_test_castlike_FLOAT_to_FLOAT8E5M2_expanded_model",
    "node_test_castlike_FLOAT_to_FLOAT8E5M2_model",
    "node_test_castlike_FLOAT_to_STRING_expanded_model",
    "node_test_castlike_FLOAT_to_STRING_model",
    "node_test_castlike_STRING_to_FLOAT_expanded_model",
    "node_test_castlike_STRING_to_FLOAT_model",
    "node_test_dequantizelinear_e4m3fn_model",
    "node_test_dequantizelinear_e4m3fn_zero_point_model",
    "node_test_dequantizelinear_e5m2_model",
    "node_test_equal_string_broadcast_model",
    "node_test_equal_string_model",
    "node_test_gru_defaults_model",
    "node_test_gru_seq_length_model",
    "node_test_gru_with_initial_bias_model",
    "node_test_identity_opt_model",
    "node_test_identity_sequence_model",
    "node_test_if_model",
    "node_test_if_opt_model",
    "node_test_if_seq_model",
    "node_test_loop11_model",
    "node_test_loop13_seq_model",
    "node_test_loop16_seq_none_model",
    "node_test_lstm_defaults_model",
    "node_test_lstm_with_initial_bias_model",
    "node_test_lstm_with_peepholes_model",
    "node_test_optional_get_element_optional_sequence_model",
    "node_test_optional_get_element_optional_tensor_model",
    "node_test_optional_get_element_sequence_model",
    "node_test_optional_has_element_empty_optional_input_model",
    "node_test_optional_has_element_optional_input_model",
    "node_test_optional_has_element_tensor_input_model",
    "node_test_quantizelinear_e4m3fn_model",
    "node_test_quantizelinear_e5m2_model",
    "node_test_range_float_type_positive_delta_expanded_model",
    "node_test_range_int32_type_negative_delta_expanded_model",
    "node_test_regex_full_match_basic_model",
    "node_test_regex_full_match_email_domain_model",
    "node_test_regex_full_match_empty_model",
    "node_test_rnn_seq_length_model",
    "node_test_scan9_sum_model",
    "node_test_scan_sum_model",
    "node_test_sequence_insert_at_back_model",
    "node_test_sequence_insert_at_front_model",
    "node_test_sequence_map_add_1_sequence_1_tensor_expanded_model",
    "node_test_sequence_map_add_1_sequence_1_tensor_model",
    "node_test_sequence_map_add_2_sequences_expanded_model",
    "node_test_sequence_map_add_2_sequences_model",
    "node_test_sequence_map_extract_shapes_expanded_model",
    "node_test_sequence_map_extract_shapes_model",
    "node_test_sequence_map_identity_1_sequence_1_tensor_expanded_model",
    "node_test_sequence_map_identity_1_sequence_1_tensor_model",
    "node_test_sequence_map_identity_1_sequence_expanded_model",
    "node_test_sequence_map_identity_1_sequence_model",
    "node_test_sequence_map_identity_2_sequences_expanded_model",
    "node_test_sequence_map_identity_2_sequences_model",
    "node_test_simple_rnn_defaults_model",
    "node_test_simple_rnn_with_initial_bias_model",
    "node_test_split_to_sequence_1_model",
    "node_test_split_to_sequence_2_model",
    "node_test_split_to_sequence_nokeepdims_model",
    "node_test_string_concat_broadcasting_model",
    "node_test_string_concat_empty_string_model",
    "node_test_string_concat_model",
    "node_test_string_concat_utf8_model",
    "node_test_string_concat_zero_dimensional_model",
    "node_test_string_split_basic_model",
    "node_test_string_split_consecutive_delimiters_model",
    "node_test_string_split_empty_string_delimiter_model",
    "node_test_string_split_empty_tensor_model",
    "node_test_string_split_maxsplit_model",
    "node_test_string_split_no_delimiter_model",
    "node_test_strnormalizer_export_monday_casesensintive_lower_model",
    "node_test_strnormalizer_export_monday_casesensintive_nochangecase_model",
    "node_test_strnormalizer_export_monday_casesensintive_upper_model",
    "node_test_strnormalizer_export_monday_empty_output_model",
    "node_test_strnormalizer_export_monday_insensintive_upper_twodim_model",
    "node_test_strnormalizer_nostopwords_nochangecase_model",
    "simple_test_sequence_model1_model",
    "simple_test_sequence_model2_model",
    "simple_test_sequence_model3_model",
    "simple_test_sequence_model4_model",
    "simple_test_sequence_model5_model",
    "simple_test_sequence_model6_model",
    "simple_test_sequence_model7_model",
    "simple_test_sequence_model8_model",
    "simple_test_strnorm_model_monday_casesensintive_lower_model",
    "simple_test_strnorm_model_monday_casesensintive_nochangecase_model",
    "simple_test_strnorm_model_monday_casesensintive_upper_model",
    "simple_test_strnorm_model_monday_empty_output_model",
    "simple_test_strnorm_model_monday_insensintive_upper_twodim_model",
    "simple_test_strnorm_model_nostopwords_nochangecase_model",
]





class ImportSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unexpected_failure_count = 0
        ImportSmokeTest.actual_failures = []

    @classmethod
    def tearDownClass(cls):
        if cls.unexpected_failure_count:
            # Print a helpful message with copy-paste XFAIL def.
            failure_report_path = OUTPUT_PATH / "import_smoke_test_report.txt"
            print(
                "Unexpected failures. Writing copy/paste report to:",
                failure_report_path,
            )
            with open(failure_report_path, "wt") as f:
                lines = [f'    "{s}",' for s in ImportSmokeTest.actual_failures]
                print(
                    f"Unexpected failures in the following. Copy/paste to update `TEST_CAST_XFAILS`:",
                    file=f,
                )
                print(f"TEST_CAST_XFAILS = [", file=f)
                [print(l, file=f) for l in lines]
                print(f"]", file=f)

        ImportSmokeTest.actual_failures.clear()

    def load_onnx_model(self, file_path: Path) -> onnx.ModelProto:
        raw_model = onnx.load(file_path)
        try:
            inferred_model = onnx.shape_inference.infer_shapes(raw_model)
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError as e:
            print("WARNING: Shape inference failure (skipping test):", e)
            self.skipTest(reason="shape inference failure")

        # inferred_model = raw_model
        return inferred_model

    def run_import_test(self, norm_name: str, rel_path: str):
        context = ir.Context()
        configure_context(context)

        model_info = onnx_importer.ModelInfo(
            self.load_onnx_model(ONNX_TEST_DATA_DIR / rel_path),
        )
        m = model_info.create_module(context=context).operation
        try:
            imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
            imp.import_all()
            m.verify()
        finally:
            # Use a ".txt" extension to avoid lit test discovery.
            with open(OUTPUT_PATH / f"{norm_name}.mlir", "wt") as f:
                print(m.get_asm(), file=f)

    def testExists(self):
        # We expect a lot of test cases. Die if not the case (i.e. if paths change
        # or something).
        self.assertGreater(len(ONNX_REL_PATHS), 10)


# Generate test methods for each onnx file.
for _rel_path in ONNX_REL_PATHS:

    def attach_test(rel_path):
        norm_name = rel_path.removesuffix(".onnx").replace("/", "_")

        def test_method(self: ImportSmokeTest):
            try:
                self.run_import_test(norm_name, rel_path)
            except onnx_importer.OnnxImportError as e:
                # All legitimate failures should be caught and reported
                # as an OnnxImportError.
                ImportSmokeTest.actual_failures.append(norm_name)
                if norm_name not in TEST_CAST_XFAILS:
                    ImportSmokeTest.unexpected_failure_count += 1
                raise e

        test_method.__name__ = f"test_{norm_name}"

        if norm_name in TEST_CAST_XFAILS:
            test_method = unittest.expectedFailure(test_method)

        setattr(ImportSmokeTest, test_method.__name__, test_method)

    attach_test(_rel_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
