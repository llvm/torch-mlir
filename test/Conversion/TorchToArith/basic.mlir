// RUN: torch-mlir-opt <%s -convert-torch-to-arith | FileCheck %s


// CHECK-LABEL:   func.func @torch.aten.dim(
// CHECK-SAME:                         %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.int {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<*,f32> -> tensor<*xf32>
// CHECK:           %[[RANK:.*]] = tensor.rank %[[BUILTIN_TENSOR]] : tensor<*xf32>
// CHECK:           %[[RANK_I64:.*]] = arith.index_cast %[[RANK]] : index to i64
// CHECK:           %[[RANK_TORCH_INT:.*]] = torch_c.from_i64 %[[RANK_I64]]
// CHECK:           return %[[RANK_TORCH_INT]] : !torch.int
func.func @torch.aten.dim(%arg0: !torch.vtensor<*,f32>) -> !torch.int {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<*,f32> -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.runtime.assert(
// CHECK-SAME:                            %[[X:.*]]: !torch.int,
// CHECK-SAME:                            %[[Y:.*]]: !torch.int) {
// CHECK-DAG:       %[[X_I64:.*]] = torch_c.to_i64 %[[X]]
// CHECK-DAG:       %[[Y_I64:.*]] = torch_c.to_i64 %[[Y]]
// CHECK:           %[[CMP:.*]] = arith.cmpi ne, %[[X_I64]], %[[Y_I64]] : i64
// CHECK:           assert %[[CMP]], "x must not be equal to y"
// CHECK:           return
func.func @torch.runtime.assert(%arg0: !torch.int, %arg1: !torch.int) {
  %0 = torch.aten.ne.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  torch.runtime.assert %0, "x must not be equal to y"
  return
}

// CHECK-LABEL:   func.func @torch.aten.ne.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi ne, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.ne.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.eq.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.eq.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.gt.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.gt.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.gt.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi sge, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.ge.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ge.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}


// CHECK-LABEL:   func.func @torch.aten.lt.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.lt.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.lt.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.le.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpi sle, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.le.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.le.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.vtensor.literal() -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VTENSOR:.*]] = torch_c.from_builtin_tensor %[[CST]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VTENSOR]] : !torch.vtensor<[],f32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[],f32> {
  %0 = torch.vtensor.literal(dense<0.0> : tensor<f32>) : !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @torch.constant.bool() -> !torch.bool {
// CHECK:           %[[CST:.*]] = arith.constant true
// CHECK:           %[[BOOL:.*]] = torch_c.from_i1 %[[CST]]
// CHECK:           return %[[BOOL]] : !torch.bool
func.func @torch.constant.bool() -> !torch.bool {
  %true = torch.constant.bool true
  return %true : !torch.bool
}

// CHECK-LABEL:   func.func @torch.constant.float() -> !torch.float {
// CHECK:           %[[CST:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[FLOAT:.*]] = torch_c.from_f64 %[[CST]]
// CHECK:           return %[[FLOAT]] : !torch.float
func.func @torch.constant.float() -> !torch.float {
  %float = torch.constant.float 1.000000e+00
  return %float : !torch.float
}

// CHECK-LABEL:  func.func @torch.constant.int() -> !torch.int {
// CHECK:          %[[CST:.*]]  = arith.constant 1 : i64
// CHECK:          %[[INT:.*]] = torch_c.from_i64 %[[CST]]
// CHECK:          return %[[INT]] : !torch.int
func.func @torch.constant.int() -> !torch.int {
  %int1 = torch.constant.int 1
  return %int1 : !torch.int
}

// CHECK-LABEL:  func.func @torch.aten.add.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.int {
// CHECK-DAG:      %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:      %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:          %[[ADD:.*]] = arith.addi %[[LHS_I64:.*]], [[RHS_I64:.*]] : i64
// CHECK:          %[[OUT:.*]] = torch_c.from_i64 %[[INT:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.int
func.func @torch.aten.add.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.add.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:  func.func @torch.aten.sub.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.int {
// CHECK-DAG:      %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:      %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:          %[[SUB:.*]] = arith.subi %[[LHS_I64:.*]], [[RHS_I64:.*]] : i64
// CHECK:          %[[OUT:.*]] = torch_c.from_i64 %[[INT:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.int
func.func @torch.aten.sub.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.sub.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:  func.func @torch.aten.sub.float(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.float) -> !torch.float {
// CHECK-DAG:      %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:      %[[RHS_F64:.*]] = torch_c.to_f64 %[[RHS]]
// CHECK:          %[[SUB:.*]] = arith.subf %[[LHS_F64:.*]], [[RHS_F64:.*]] : f64
// CHECK:          %[[OUT:.*]] = torch_c.from_f64 %[[SUB:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.float
func.func @torch.aten.sub.float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.float {
  %0 = torch.aten.sub.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:  func.func @torch.aten.mul.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.int {
// CHECK-DAG:      %[[LHS_I64:.*]] = torch_c.to_i64 %[[LHS]]
// CHECK-DAG:      %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:          %[[MUL:.*]] = arith.muli %[[LHS_I64:.*]], [[RHS_I64:.*]] : i64
// CHECK:          %[[OUT:.*]] = torch_c.from_i64 %[[MUL:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.int
func.func @torch.aten.mul.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.mul.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:  func.func @torch.aten.mul.float_int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.float {
// CHECK-DAG:      %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:      %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:          %[[MUL:.*]] = arith.mulf %[[LHS_F64:.*]], [[RHS_I64:.*]] : f64
// CHECK:          %[[OUT:.*]] = torch_c.from_f64 %[[MUL:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.float
func.func @torch.aten.mul.float_int(%arg0: !torch.float, %arg1: !torch.int) -> !torch.float {
  %0 = torch.aten.mul.float_int %arg0, %arg1 : !torch.float, !torch.int -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:  func.func @torch.aten.div.float(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.float) -> !torch.float {
// CHECK-DAG:      %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:      %[[RHS_F64:.*]] = torch_c.to_f64 %[[RHS]]
// CHECK:          %[[SUB:.*]] = arith.divf %[[LHS_F64:.*]], [[RHS_F64:.*]] : f64
// CHECK:          %[[OUT:.*]] = torch_c.from_f64 %[[SUB:.*]]
// CHECK:          return %[[OUT:.*]] : !torch.float
func.func @torch.aten.div.float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.float {
  %0 = torch.aten.div.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.ge.float(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.float) -> !torch.bool {
// CHECK-DAG:       %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:       %[[RHS_F64:.*]] = torch_c.to_f64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = arith.cmpf uge, %[[LHS_F64]], %[[RHS_F64]] : f64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.ge.float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.bool {
  %0 = torch.aten.ge.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.float_int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[RHS_F64:.*]] = arith.sitofp %[[RHS_I64]] : i64 to f64
// CHECK:           %[[CMP:.*]] = arith.cmpf uge, %[[LHS_F64]], %[[RHS_F64]] : f64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.ge.float_int(%arg0: !torch.float, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ge.float_int %arg0, %arg1 : !torch.float, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.float_int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[RHS_F64:.*]] = arith.sitofp %[[RHS_I64]] : i64 to f64
// CHECK:           %[[CMP:.*]] = arith.cmpf une, %[[LHS_F64]], %[[RHS_F64]] : f64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.ne.float_int(%arg0: !torch.float, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.float_int %arg0, %arg1 : !torch.float, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:  func.func @torch.aten.ceil.float(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.float) -> !torch.int {
// CHECK:          %[[ARG_F64:.*]] = torch_c.to_f64 %[[ARG]]
// CHECK:          %[[CEIL:.*]] = math.ceil %[[ARG_F64]] : f64
// CHECK:          %[[CEIL_I64:.*]] = arith.fptosi %[[CEIL]] : f64 to i64
// CHECK:          %[[OUT:.*]] = torch_c.from_i64 %[[CEIL_I64]]
// CHECK:          return %[[OUT]] : !torch.int
func.func @torch.aten.ceil.float(%arg0: !torch.float) -> !torch.int {
  %0 = torch.aten.ceil.float %arg0 : !torch.float -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.gt.float_int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.float,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK-DAG:       %[[LHS_F64:.*]] = torch_c.to_f64 %[[LHS]]
// CHECK-DAG:       %[[RHS_I64:.*]] = torch_c.to_i64 %[[RHS]]
// CHECK:           %[[RHS_F64:.*]] = arith.sitofp %[[RHS_I64]] : i64 to f64
// CHECK:           %[[CMP:.*]] = arith.cmpf ugt, %[[LHS_F64]], %[[RHS_F64]] : f64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func.func @torch.aten.gt.float_int(%arg0: !torch.float, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.gt.float_int %arg0, %arg1 : !torch.float, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.sqrt.int(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.int) -> !torch.float {
// CHECK:           %[[ARG_I64:.*]] = torch_c.to_i64 %[[ARG]]
// CHECK:           %[[ARG_F64:.*]] = arith.sitofp %[[ARG_I64]] : i64 to f64
// CHECK:           %[[SQRT:.*]] = math.sqrt %[[ARG_F64]] : f64
// CHECK:           %[[SQRT_TORCH_FLOAT:.*]] = torch_c.from_f64 %[[SQRT]]
// CHECK:           return %[[SQRT_TORCH_FLOAT]] : !torch.float
func.func @torch.aten.sqrt.int(%arg0: !torch.int) -> !torch.float {
  %0 = torch.aten.sqrt.int %arg0 : !torch.int -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.Bool.float(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.float) -> !torch.bool {
// CHECK:           %[[ARG_F64:.*]] = torch_c.to_f64 %[[ARG]]
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[CMP:.*]] = arith.cmpf une, %[[ARG_F64]], %[[CST]] : f64
// CHECK:           %[[SELECT:.*]] = arith.select %[[CMP]], %[[TRUE]], %[[FALSE]] : i1
// CHECK:           %[[OUT:.*]] = torch_c.from_i1 %[[SELECT]]
// CHECK:           return %[[OUT]] : !torch.bool
func.func @torch.aten.Bool.float(%arg0: !torch.float) -> !torch.bool {
  %0 = torch.aten.Bool.float %arg0 : !torch.float -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.Bool.int(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.int) -> !torch.bool {
// CHECK:           %[[ARG_I64:.*]] = torch_c.to_i64 %[[ARG]]
// CHECK:           %[[CST:.*]] = arith.constant 0 : i64
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[CMP:.*]] = arith.cmpi ne, %[[ARG_I64]], %[[CST]] : i64
// CHECK:           %[[SELECT:.*]] = arith.select %[[CMP]], %[[TRUE]], %[[FALSE]] : i1
// CHECK:           %[[OUT:.*]] = torch_c.from_i1 %[[SELECT]]
// CHECK:           return %[[OUT]] : !torch.bool
func.func @torch.aten.Bool.int(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.Bool.int %arg0 : !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.Int.bool(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.bool) -> !torch.int {
// CHECK:           %[[ARG_I1:.*]] = torch_c.to_i1 %[[ARG]]
// CHECK:           %[[EXTUI:.*]] = arith.extui %[[ARG_I1]] : i1 to i64
// CHECK:           %[[OUT:.*]] = torch_c.from_i64 %[[EXTUI]]
// CHECK:           return %[[OUT]] : !torch.int
func.func @torch.aten.Int.bool(%arg0: !torch.bool) -> !torch.int {
  %0 = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.Int.Scalar(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.float) -> !torch.int {
// CHECK:         %[[ARG_F64:.*]] = torch_c.to_f64 %[[ARG]]
// CHECK:         %[[FPTOSI:.*]] = arith.fptosi %[[ARG_F64]] : f64 to i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[FPTOSI]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @torch.aten.Int.Scalar(%arg0: !torch.float) -> !torch.int {
  %0 = torch.aten.Int.Scalar %arg0 : !torch.float -> !torch.int
  return %0 : !torch.int
}
