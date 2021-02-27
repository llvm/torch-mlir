//===- graph_importer.cpp -------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "graph_importer.h"

#include <unordered_map>

#include "mlir_utils.h"
#include "torch_to_mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"

namespace py = pybind11;
using namespace torch_mlir;

MlirOperation torch_mlir::importGraphAsFuncOp(MlirContext context,
                                              torch::jit::Graph *graph,
                                              const std::string &name) {
  // Useful for debugging:
  // graph->dump();
  MlirLocation loc = mlirLocationUnknownGet(context);
  MlirAttribute typeAttr =
      mlirTypeAttrGet(getFunctionTypeFromBlock(context, graph->block()));
  MlirAttribute symNameAttr = mlirStringAttrGet(context, toMlirStringRef(name));
  MlirOperation func = createMlirOperation(
      "func", loc, mlirRegionCreate(), toMlirNamedAttribute("type", typeAttr),
      toMlirNamedAttribute("sym_name", symNameAttr));
  MlirRegion bodyRegion = mlirOperationGetRegion(func, 0);
  MlirBlock block = importBlock(context, graph->block(), "std.return");
  mlirRegionAppendOwnedBlock(bodyRegion, block);
  return func;
}
