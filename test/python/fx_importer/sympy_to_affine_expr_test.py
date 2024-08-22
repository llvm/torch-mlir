# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s
# This file contains tests checking translating sympy expressions to (semi-)affine expressions.

from sympy import Symbol
from torch_mlir.extras.fx_importer import sympy_expr_to_semi_affine_expr

from torch_mlir.ir import (
    AffineSymbolExpr,
    Context,
)


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_sympy_to_semi_affine_expr_translation
def test_sympy_to_semi_affine_expr_translation():
    with Context():
        s0 = Symbol("s0", positive=True, integer=True)
        s1 = Symbol("s1", positive=True, integer=True)

        symbols_set = sorted({s0, s1}, key=lambda x: x.name)
        symbols_map = {
            str(symbol): AffineSymbolExpr.get(i) for i, symbol in enumerate(symbols_set)
        }

        SYMPY_EXPRS = [
            # CHECK: 10
            (10),
            # CHECK: s0
            (s0),
            # CHECK: s0
            (s0 + 0),
            # CHECK: s0 + 1
            (s0 + 1),
            # CHECK: s0
            (s0 * 1),
            # CHECK: s0 * 2
            (s0 * 2),
            # CHECK: s0 * s0
            (s0 * s0),
            # CHECK: s0 * s1
            (s0 * s1),
            # CHECK: s0 * s0
            (s0**2),
            # CHECK: (s0 * s0) * s0
            (s0**3),
            # CHECK: ((((s0 * s0) * s0) * s0) * s0) * s0
            ((s0**2) ** 3),
            # CHECK: ((((((s0 * s0) * s0) * s0) * s0) * s0) * s0) * s0
            (s0 ** (2**3)),
            # CHECK: s0 mod 10
            (s0 % 10),
            # CHECK: s0 - s1 * 2 + 5
            (s0 + 5 - 2 * s1),
        ]

        for expr in SYMPY_EXPRS:
            print(sympy_expr_to_semi_affine_expr(expr, symbols_map))
