//===- ATenOpInterface.td ----------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

#ifndef NPCOMP_DIALECT_ATEN_IR_ATEN_OP_INTERFACES
#define NPCOMP_DIALECT_ATEN_IR_ATEN_OP_INTERFACES

def StatisticsOpInterface : OpInterface<"StatisticsOpInterface"> {
  let description = [{
     This interface allows ops to expose a static operation profile,
     describing the computational behavior of their function.
  }];

  let methods = [
    InterfaceMethod<[{
      Return statistics about the compute requirements of an op.  The return
      value maps an arbitrary set of statistic names to an integer value.
      Users are currently expected to accept any statistic names and statistic
      names are arbitrary for different operations.  In the future this
      interface could be expanded and standardized.
      }],
      "std::map<std::string, uint64_t>", "getStatistics"
    >,

    InterfaceMethod<[{
      Return memory transfer requirements of the operation for the
      operand with the given index.
      }],
      "uint64_t", "getOperandTransferVolume",
      (ins "unsigned int":$idx, "bool":$read), /*methodBody=*/[{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        if (!read) return 0;
        Value v = op->getODSOperands(idx).front();
        auto ty = v.getType();
        return mlir::NPCOMP::aten::getTensorVolume(ty);
    }]>,

    InterfaceMethod<[{
      Return memory transfer requirements of the operation for the
      result with the given index.
      }],
      "uint64_t", "getResultTransferVolume",
      (ins "unsigned int":$idx, "bool":$write), /*methodBody=*/[{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        if (!write) return 0;
        Value v = op->getODSResults(idx).front();
        auto ty = v.getType();
        return mlir::NPCOMP::aten::getTensorVolume(ty);
    }]>,

  ];
}

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                       "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;


#endif // NPCOMP_DIALECT_ATEN_IR_ATEN_OP_INTERFACES
