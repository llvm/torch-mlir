//===- python_bindings.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "../pybind.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include "../init_python_bindings.h"
#include "acap_dispatch.h"
#include "module_builder.h"

using namespace torch_mlir;
namespace py = pybind11;

namespace {

static const char kGetRegisteredOpsDocstring[] =
    R"(Gets a data structure of all registered ops.

The returned data reflects the metadata available in the c10 dispatcher at
the time of this call. It is meant for various code generation tools.

Returns:
  A list of records, one for each op. Each record is a dict of the following:
    "name": tuple -> (qualified_name, overload)
    "is_vararg": bool -> Whether the op accepts variable arguments
    "is_varret": bool -> Whether the op produces variable returns
    "arguments" and "returns": List[Dict] -> Having keys:
      "type": str -> PyTorch type name as in op signatures
      "pytype": str -> PyType style type annotation
      "N": (optional) int -> For list types, the arity
      "default_debug": (optional) str -> Debug printout of the default value
      "alias_info": Dict -> Alias info with keys "before" and "after"
)";

class LambdaOpRegistrationListener : public c10::OpRegistrationListener {
public:
  using CallbackTy = std::function<void(const c10::OperatorHandle &)>;
  LambdaOpRegistrationListener(CallbackTy callback)
      : callback(std::move(callback)) {}
  void onOperatorRegistered(const c10::OperatorHandle &op) override {
    callback(op);
  }
  void onOperatorDeregistered(const c10::OperatorHandle &op) override {}

private:
  CallbackTy callback;
};

py::list GetRegisteredOps() {
  py::list results;
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  auto listener = std::make_unique<LambdaOpRegistrationListener>(
      [&](const c10::OperatorHandle &op) -> void {
        if (!op.hasSchema()) {
          // Legacy?
          return;
        }

        py::dict record;
        {
          py::tuple name(2);
          name[0] = op.operator_name().name;
          name[1] = op.operator_name().overload_name;
          record["name"] = std::move(name);
        }

        auto &schema = op.schema();
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
      });
  // Note: addRegistrationListener reports all currently registered ops
  // during the call and then incrementally reports newer ops until the RAII
  // return value is destroyed. Since we only want the current, surround in
  // a block so it immediately unregisters.
  { dispatcher.addRegistrationListener(std::move(listener)); }
  return results;
}

void InitModuleBindings(py::module &m) {
  py::class_<AcapController, std::shared_ptr<AcapController>>(m,
                                                              "AcapController")
      .def("__enter__", &AcapController::contextEnter)
      .def("__exit__", &AcapController::contextExit)
      .def("get_debug_log", &AcapController::getDebugLog);
  m.def("get_registered_ops", &GetRegisteredOps, kGetRegisteredOpsDocstring);

  ModuleBuilder::bind(m);
}

} // namespace

void torch_mlir::InitC10DispatchBindings(py::module &m) {
  InitModuleBindings(m);
}
