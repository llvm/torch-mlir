//===- get_registered_ops.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "get_registered_ops.h"

#include <torch/csrc/jit/ir/ir.h>

namespace py = pybind11;

static const char kGetRegisteredOpsDocstring[] =
    R"(Gets a data structure of all registered ops.

The returned data reflects the metadata available in the Torch JIT's
registry at the time of this call. It includes both the operators available
in the c10 dispatcher and an auxiliary set of operators that the Torch JIT
uses to implement auxiliary operations that in the non-TorchScript case
are performed by Python itself.

This information is meant for various code generation tools.

Returns:
  A list of records, one for each `torch::jit::Operator`. Known to the
  Torch JIT operator registry. Each record is a dict of the following:
    "name": tuple -> (qualified_name, overload)
    "is_c10_op": bool -> Whether the op is in the c10 dispatcher registry,
                         or is a JIT-only op.
    "is_vararg": bool -> Whether the op accepts variable arguments
    "is_varret": bool -> Whether the op produces variable returns
    "is_mutable": bool -> Whether the op potentially mutates any operand
    "arguments" and "returns": List[Dict] -> Having keys:
      "type": str -> PyTorch type name as in op signatures
      "pytype": str -> PyType style type annotation
      "N": (optional) int -> For list types, the arity
      "default_debug": (optional) str -> Debug printout of the default value
      "alias_info": Dict -> Alias info with keys "before" and "after"
)";

static py::list getRegisteredOps() {
  py::list results;

  // Walk the JIT operator registry to find all the ops that we might need
  // for introspection / ODS generation.
  // This registry contains a superset of the ops available to the dispatcher,
  // since the JIT has its own dispatch mechanism that it uses to implement
  // "prim" ops and a handful of "aten" ops that are effectively prim ops, such
  // as `aten::__is__`.
  for (const std::shared_ptr<torch::jit::Operator> &op :
       torch::jit::getAllOperators()) {
    const c10::FunctionSchema &schema = op->schema();

    py::dict record;
    {
      py::tuple name(2);
      name[0] = schema.name();
      name[1] = schema.overload_name();
      record["name"] = std::move(name);
    }

    record["is_c10_op"] = op->isC10Op();
    record["is_vararg"] = schema.is_vararg();
    record["is_varret"] = schema.is_varret();
    record["is_mutable"] = schema.is_mutable();

    py::list arguments;
    py::list returns;
    auto addArgument = [](py::list &container, const c10::Argument &arg) {
      py::dict argRecord;
      argRecord["name"] = arg.name();
      argRecord["type"] = arg.type()->str();
      argRecord["pytype"] = arg.type()->annotation_str();
      if (arg.N())
        argRecord["N"] = *arg.N();
      // TODO: If the default value becomes useful, switch on it and return
      // a real value, not just a string print.
      if (arg.default_value()) {
        std::stringstream sout;
        sout << *arg.default_value();
        argRecord["default_debug"] = sout.str();
      }
      if (arg.alias_info()) {
        py::dict aliasInfo;
        py::list before;
        py::list after;
        for (auto &symbol : arg.alias_info()->beforeSets()) {
          before.append(std::string(symbol.toQualString()));
        }
        for (auto &symbol : arg.alias_info()->afterSets()) {
          after.append(std::string(symbol.toQualString()));
        }
        aliasInfo["is_write"] = arg.alias_info()->isWrite();
        aliasInfo["before"] = std::move(before);
        aliasInfo["after"] = std::move(after);
        argRecord["alias_info"] = std::move(aliasInfo);
      }

      container.append(std::move(argRecord));
    };
    for (auto &argument : schema.arguments()) {
      addArgument(arguments, argument);
    }
    for (auto &returnArg : schema.returns()) {
      addArgument(returns, returnArg);
    }
    record["arguments"] = std::move(arguments);
    record["returns"] = std::move(returns);
    results.append(std::move(record));
  }

  return results;
}

void torch_mlir::initGetRegisteredOpsBindings(py::module &m) {
  m.def("get_registered_ops", &getRegisteredOps, kGetRegisteredOpsDocstring);
}
