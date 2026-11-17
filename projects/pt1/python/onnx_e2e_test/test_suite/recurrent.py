# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Recurrent ONNX op tests (LSTM, GRU, RNN)."""

import numpy as np
import torch
import onnx.helper
import onnx.numpy_helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# LSTM — single-direction; seq=5, batch=1, input_size=4, hidden_size=8.
# X is the only runtime input; W/R/B are baked in as initializers.
#   Outputs: Y [seq, num_dir, batch, H] and Y_h [num_dir, batch, H].
# ==============================================================================


@register_onnx_test
class OnnxLSTM_forward_basic(OnnxTestCase):
    _SEQ, _BATCH, _INPUT, _HIDDEN = 5, 1, 4, 8
    _NUM_DIR = 1

    def __init__(self):
        rng = np.random.default_rng(7)
        self._W = rng.standard_normal(
            (self._NUM_DIR, 4 * self._HIDDEN, self._INPUT)
        ).astype(np.float32)
        self._R = rng.standard_normal(
            (self._NUM_DIR, 4 * self._HIDDEN, self._HIDDEN)
        ).astype(np.float32)
        self._B = rng.standard_normal((self._NUM_DIR, 8 * self._HIDDEN)).astype(
            np.float32
        )

    @annotate_inputs([("X", torch.float32, [_SEQ, _BATCH, _INPUT])])
    def graph(self):
        node = onnx.helper.make_node(
            "LSTM",
            ["X", "W", "R", "B"],
            ["Y", "Y_h"],
            hidden_size=self._HIDDEN,
            direction="forward",
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="W"),
            onnx.numpy_helper.from_array(self._R, name="R"),
            onnx.numpy_helper.from_array(self._B, name="B"),
        ]
        return [node], ["Y", "Y_h"], initializers


# ==============================================================================
# Gate activations with activation_alpha / activation_beta.
# seq=4, batch=2, input_size=3, hidden_size=3; X is the runtime input and
# W/R/B are initializers. onnx.reference ignores the activations of LSTM and
# GRU and supports only Tanh and Affine for RNN, so these tests use ONNX
# Runtime as the reference.
# ==============================================================================

_GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


class _RecurrentActivationTest(OnnxTestCase):
    use_onnxruntime_reference = True
    _SEQ, _BATCH, _INPUT, _HIDDEN = 4, 2, 3, 3
    _OP = "RNN"
    _DIRECTION = "forward"
    _ACTIVATIONS: list = []
    _ALPHA: list = []
    _BETA: list = []

    def _weights(self, num_dir, gates):
        rng = np.random.default_rng(11)
        w = rng.standard_normal((num_dir, gates * self._HIDDEN, self._INPUT))
        r = rng.standard_normal((num_dir, gates * self._HIDDEN, self._HIDDEN))
        b = rng.standard_normal((num_dir, 2 * gates * self._HIDDEN))
        return [a.astype(np.float32) for a in (w, r, b)]

    @annotate_inputs([("X", torch.float32, [_SEQ, _BATCH, _INPUT])])
    def graph(self):
        num_dir = 2 if self._DIRECTION == "bidirectional" else 1
        w, r, b = self._weights(num_dir, _GATES[self._OP])
        attrs = {"hidden_size": self._HIDDEN, "direction": self._DIRECTION}
        if self._ACTIVATIONS:
            attrs["activations"] = self._ACTIVATIONS
        if self._ALPHA:
            attrs["activation_alpha"] = self._ALPHA
        if self._BETA:
            attrs["activation_beta"] = self._BETA
        if self._OP == "GRU":
            attrs["linear_before_reset"] = 1
        node = onnx.helper.make_node(
            self._OP, ["X", "W", "R", "B"], ["Y", "Y_h"], **attrs
        )
        initializers = [
            onnx.numpy_helper.from_array(w, name="W"),
            onnx.numpy_helper.from_array(r, name="R"),
            onnx.numpy_helper.from_array(b, name="B"),
        ]
        return [node], ["Y", "Y_h"], initializers


# HardSigmoid gates with their default alpha and beta, as tf2onnx emits for
# Keras LSTMs with recurrent_activation="hard_sigmoid".
@register_onnx_test
class OnnxLSTM_hardsigmoid_gates(_RecurrentActivationTest):
    _OP = "LSTM"
    _ACTIVATIONS = ["HardSigmoid", "Tanh", "Tanh"]


@register_onnx_test
class OnnxGRU_hardsigmoid_gates(_RecurrentActivationTest):
    _OP = "GRU"
    _ACTIVATIONS = ["HardSigmoid", "Tanh"]


# Each activation through RNN, with explicit parameters where it takes them.
@register_onnx_test
class OnnxRNN_relu(_RecurrentActivationTest):
    _ACTIVATIONS = ["Relu"]


@register_onnx_test
class OnnxRNN_sigmoid(_RecurrentActivationTest):
    _ACTIVATIONS = ["Sigmoid"]


@register_onnx_test
class OnnxRNN_affine(_RecurrentActivationTest):
    _ACTIVATIONS = ["Affine"]
    _ALPHA = [0.5]
    _BETA = [0.25]


@register_onnx_test
class OnnxRNN_leakyrelu(_RecurrentActivationTest):
    _ACTIVATIONS = ["LeakyRelu"]
    _ALPHA = [0.1]


@register_onnx_test
class OnnxRNN_thresholdedrelu(_RecurrentActivationTest):
    _ACTIVATIONS = ["ThresholdedRelu"]
    _ALPHA = [0.5]


@register_onnx_test
class OnnxRNN_scaledtanh(_RecurrentActivationTest):
    _ACTIVATIONS = ["ScaledTanh"]
    _ALPHA = [1.5]
    _BETA = [0.5]


@register_onnx_test
class OnnxRNN_hardsigmoid(_RecurrentActivationTest):
    _ACTIVATIONS = ["HardSigmoid"]
    _ALPHA = [0.3]
    _BETA = [0.4]


@register_onnx_test
class OnnxRNN_elu(_RecurrentActivationTest):
    _ACTIVATIONS = ["Elu"]
    _ALPHA = [1.5]


@register_onnx_test
class OnnxRNN_softsign(_RecurrentActivationTest):
    _ACTIVATIONS = ["Softsign"]


@register_onnx_test
class OnnxRNN_softplus(_RecurrentActivationTest):
    _ACTIVATIONS = ["Softplus"]


# Affine with only one of alpha and beta: the other is 0, as in ONNX Runtime.
@register_onnx_test
class OnnxRNN_affine_alpha_only(_RecurrentActivationTest):
    _ACTIVATIONS = ["Affine"]
    _ALPHA = [0.5]


@register_onnx_test
class OnnxRNN_affine_beta_only(_RecurrentActivationTest):
    _ACTIVATIONS = ["Affine"]
    _BETA = [0.25]


# Affine and ScaledTanh with neither alpha nor beta: both are 0, as in ONNX
# Runtime.
@register_onnx_test
class OnnxRNN_affine_no_parameters(_RecurrentActivationTest):
    _ACTIVATIONS = ["Affine"]


@register_onnx_test
class OnnxRNN_scaledtanh_no_parameters(_RecurrentActivationTest):
    _ACTIVATIONS = ["ScaledTanh"]


# activation_alpha is consumed only by activations that take alpha: LeakyRelu
# (h) gets the second value because Tanh (g) takes none.
@register_onnx_test
class OnnxLSTM_alpha_order(_RecurrentActivationTest):
    _OP = "LSTM"
    _ACTIVATIONS = ["HardSigmoid", "Tanh", "LeakyRelu"]
    _ALPHA = [0.3, 0.2]
    _BETA = [0.4]


# For a bidirectional LSTM the parameter lists continue across all six
# activations, so the reverse f (LeakyRelu) gets the second alpha.
@register_onnx_test
class OnnxLSTM_bidirectional_reverse_alpha(_RecurrentActivationTest):
    _OP = "LSTM"
    _DIRECTION = "bidirectional"
    _ACTIVATIONS = ["HardSigmoid", "Tanh", "Tanh", "LeakyRelu", "Tanh", "Tanh"]
    _ALPHA = [0.3, 0.2]
    _BETA = [0.4]


# ThresholdedRelu keeps x only where x > alpha. With zero weights and an input
# bias of alpha, every pre-activation equals alpha, so every output is 0.
@register_onnx_test
class OnnxRNN_thresholdedrelu_at_alpha(_RecurrentActivationTest):
    _ACTIVATIONS = ["ThresholdedRelu"]
    _ALPHA = [0.5]

    def _weights(self, num_dir, gates):
        shape = (num_dir, gates * self._HIDDEN)
        w = np.zeros((*shape, self._INPUT), np.float32)
        r = np.zeros((*shape, self._HIDDEN), np.float32)
        b = np.concatenate(
            [np.full(shape, 0.5, np.float32), np.zeros(shape, np.float32)], axis=1
        )
        return [w, r, b]
