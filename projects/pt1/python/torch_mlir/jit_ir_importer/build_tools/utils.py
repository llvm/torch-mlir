# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import TextIO

from contextlib import contextmanager
import textwrap

class TextEmitter:
    """Helper for emitting text files"""
    _INDENT = "  "

    def __init__(self, out: TextIO):
        super().__init__()
        self.out = out
        self.indent_level = 0

    @contextmanager
    def indent(self, level: int = 1):
        self.indent_level += level
        yield
        self.indent_level -= level
        assert self.indent_level >= 0, "Unbalanced indentation"

    def print(self, s: str):
        current_indent = self._INDENT * self.indent_level
        for line in s.splitlines():
            self.out.write(current_indent + line + "\n")

    def quote(self, s: str) -> str:
        s = s.replace(r'"', r'\\"')
        return f'"{s}"'

    def quote_multiline_docstring(self, s: str, indent_level: int = 0) -> str:
        # TODO: Possibly find a python module to markdown the docstring for
        # better document generation.
        # Unlikely to contain the delimiter and since just a docstring, be safe.
        s = s.replace("}]", "")
        # Strip each line.
        s = "\n".join([l.rstrip() for l in s.splitlines()])
        indent = self._INDENT * indent_level
        s = textwrap.indent(s, indent + self._INDENT)
        return "[{\n" + s + "\n" + indent + "}]"
