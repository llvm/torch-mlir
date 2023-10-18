// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-linalg -split-input-file | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @tanh(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[TANH:.*]] = math.tanh %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[TANH]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @tanh(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @clamp(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CST0:.*]] = arith.constant 1.000000e-01 : f32
// CHECK:           %[[MAX:.*]] = arith.maximumf %[[BBARG0]], %[[CST0]] : f32
// CHECK:           %[[CST1:.*]] = arith.constant 1.024000e+03 : f32
// CHECK:           %[[MIN:.*]] = arith.minimumf %[[MAX]], %[[CST1]] : f32
// CHECK:           linalg.yield %[[MIN]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @clamp(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.clamp %arg0 {max_float = 1.024000e+03 : f32, min_float = 1.000000e-01 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @sigmoid(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CONST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[NEG:.*]] = arith.negf %[[IN]] : f32
// CHECK:           %[[EXP:.*]] = math.exp %[[NEG]] : f32
// CHECK:           %[[ADD:.*]] = arith.addf %[[EXP]], %[[CONST]] : f32
// CHECK:           %[[DIV:.*]] = arith.divf %[[CONST]], %[[ADD]] : f32
// CHECK:           linalg.yield %[[DIV]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @sigmoid(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sigmoid %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @sqrt(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[SQRT:.*]] = math.sqrt %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[SQRT]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @sqrt(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sqrt %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @ceil(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.ceil %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @ceil(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.ceil %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @floor(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.floor %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @floor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.floor %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @sin(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.sin %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @sin(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sin %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @cos(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.cos %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @cos(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.cos %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @abs(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.absf %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @abs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.abs %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @abs(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xi32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xi32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xi32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xi32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: i32, %{{.*}}: i32):
// CHECK:           %[[CEIL:.*]] = math.absi %[[BBARG0]] : i32
// CHECK:           linalg.yield %[[CEIL]] : i32
// CHECK:         } -> tensor<?x?xi32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xi32>
// CHECK:       }
func.func @abs(%arg0 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = tcp.abs %arg0 : tensor<?x?xi32> -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @log(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.log %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @log(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.log %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @neg(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = arith.negf %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @neg(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.neg %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @atan(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CEIL:.*]] = math.atan %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[CEIL]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @atan(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.atan %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @cast_i1(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xi32>) -> tensor<?x?xi1> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xi32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xi32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi1>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xi32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xi1>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: i32, %{{.*}}: i1):
// CHECK:           %[[CSTZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[RESULT:.*]] = arith.cmpi ne, %in, %[[CSTZERO]] : i32
// CHECK:           linalg.yield %[[RESULT]] : i1
// CHECK:         } -> tensor<?x?xi1>
// CHECK:         return %[[GENERIC]] : tensor<?x?xi1>
// CHECK:       }
func.func @cast_i1(%arg0 : tensor<?x?xi32>) -> tensor<?x?xi1> {
  %0 = tcp.cast %arg0 {in_int_signedness = #tcp<signedness Signed>, out_int_signedness = #tcp<signedness Signless>} : tensor<?x?xi32> -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @cast_si8_f32(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xi8>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xi8>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xi8>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xi8>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: i8, %{{.*}}: f32):
// CHECK:           %[[RESULT:.*]] =  arith.sitofp %[[BBARG0]] : i8 to f32
// CHECK:           linalg.yield %[[RESULT]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @cast_si8_f32(%arg0 : tensor<?x?xi8>) -> tensor<?x?xf32> {
  %0 = tcp.cast %arg0 {in_int_signedness = #tcp<signedness Signed>} : tensor<?x?xi8> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @cast_si8_ui32(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xi8>) -> tensor<?x?xi32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xi8>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xi8>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xi8>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xi32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: i8, %{{.*}}: i32):
// CHECK:           %[[RESULT:.*]] =  arith.extsi %[[BBARG0]] : i8 to i32
// CHECK:           linalg.yield %[[RESULT]] : i32
// CHECK:         } -> tensor<?x?xi32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xi32>
// CHECK:       }
func.func @cast_si8_ui32(%arg0 : tensor<?x?xi8>) -> tensor<?x?xi32> {
  %0 = tcp.cast %arg0 {in_int_signedness = #tcp<signedness Signed>, out_int_signedness = #tcp<signedness Unsigned>} : tensor<?x?xi8> -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
