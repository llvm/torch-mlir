# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Listing of the JIT operator registry, for use in generating the `torch` dialect.
"""


import torch
import torch._C
import pybind11

def get_registered_ops():
    results = []

    # Walk the JIT operator registry to find all the ops that we might need
    # for introspection / ODS generation.
    # This registry contains a superset of the ops available to the dispatcher,
    # since the JIT has its own dispatch mechanism that it uses to implement
    # "prim" ops and a handful of "aten" ops that are effectively prim ops, such
    # as `aten::__is__`.
    for schema in torch._C._jit_get_all_schemas():
        record = {}

        record["name"] = schema.name
        record["overload_name"] = schema.overload_name
        record["is_mutable"] = schema.is_mutable

        arguments = []
        returns = []

        def add_argument(container, arg):
            arg_record = {
                "name": arg.name,
                "type": arg.type.annotation_str,
                "kwarg_only" : arg.kwarg_only,
                "is_out": arg.is_out,
            }
            if arg.default_value:
                arg_record["default_value"] = arg.default_value
            if arg.alias_info:
                alias_info = {
                    "is_write": arg.alias_info.is_write,
                    "before_set": [str(symbol) for symbol in arg.alias_info.before_set],
                    "after_set": [str(symbol) for symbol in arg.alias_info.after_set],
                }
                arg_record["alias_info"] = alias_info

            container.append(arg_record)

        for argument in schema.arguments:
            add_argument(arguments, argument)
        for return_arg in schema.returns:
            add_argument(returns, return_arg)

        record["arguments"] = arguments
        record["returns"] = returns
        results.append(record)

    return results