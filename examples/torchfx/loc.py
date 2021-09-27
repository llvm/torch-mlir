# -*- Python -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# pylint: disable=no-member, no-name-in-module, invalid-name, missing-function-docstring, fixme

from typing import Mapping
import inspect
import ast
import torch.fx

class Annotation:
    def __init__(self, name: str, row: int, col: int):
        self.name = name
        self.row = row
        self.col = col

Annotations = Mapping[torch.fx.Node, Annotation]

class LocInspector:
     #TODO: type of module?
    def __init__(self, graph: torch.fx.Graph, module: torch.nn.Module):
        self.annotations = {}
        self.graph = graph
        self.module = module
        module_lines, self.module_start_lineno = \
            inspect.getsourcelines(type(module))
        module_src = "".join(module_lines)
        self.src_file = inspect.getsourcefile(type(module))
        self.module_ast = ast.parse(module_src)

    def __str__(self):
        newline = "\n\n"
        values = ["Annotations: ", str(self.annotations), newline,
                  "Src File: ", self.src_file, newline,
                  "Module AST: ", ast.dump(self.module_ast)]
        return "".join(values)

    def annotate_defs(self) -> None:
        for node in ast.walk(self.module_ast):
            if isinstance(node, (ast.ClassDef,
                                 ast.FunctionDef)):
                # subtract 1 because lineno's begin on 1
                lineno = node.lineno + self.module_start_lineno - 1
                self.annotations[node.name] = (self.src_file, lineno,
                                               node.col_offset)
