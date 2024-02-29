// RUN: torch-mlir-opt -split-input-file -verify-diagnostics %s

func.func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0, 1, 2>} unique_indices(true)
      ins(%update, %indices : memref<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        tm_tensor.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_mixed_tensor_memref(
    %update : tensor<?x?xf32>, %indices : memref<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0, 1, 2>} unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, memref<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        tm_tensor.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_extra_outputs(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error @+1 {{expected number of outputs to be same as the number of results}}
  %0, %1 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        tm_tensor.yield %1 : f32
      } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

func.func @scatter_mixed_tensor_memref(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : memref<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : memref<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        tm_tensor.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_output_type_mismatch(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<4x?xf32> {
  // expected-error @+1 {{expected type of `outs` operand #0 'tensor<?x?xf32>' to be same as result type 'tensor<4x?xf32>'}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        tm_tensor.yield %1 : f32
      } -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----

func.func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : memref<?x?xf32>) {
  // expected-error @+1 {{expected inputs and outputs to be MemRefType or scalar}}
  tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : memref<?x?xf32>, tensor<?x1xi32>)
    outs(%original : memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    }
  return
}

// -----

func.func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : memref<?x1xi32>,
    %original : tensor<?x?xf32>) {
  // expected-error @+1 {{expected inputs and outputs to be MemRefType or scalar}}
  tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    }
  return
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<64x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : tensor<64x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x?x?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{op update value rank exceeds the rank of the original value}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : tensor<?x?x?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x4xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x3xf32>) -> tensor<?x3xf32> {
  // expected-error @+1 {{shape of update value dim#1 exceeds original value at dim#1}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>} unique_indices(true)
    ins(%update, %indices : tensor<?x4xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tm_tensor.yield %1 : f32
    } -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{expected region to have scalar argument of integer or float types}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: index, %arg2: index):
      %1 = arith.addi %arg1, %arg2 : index
      %2 = arith.index_cast %1 : index to i32
      tm_tensor.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 0 of region 'i64' and element type of update value 'i32'}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %1 = arith.trunci %arg1 : i64 to i32
      %2 = arith.addi %1, %arg2 : i32
      tm_tensor.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 1 of region 'i64' and element type of original value 'i32'}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.trunci %arg2 : i64 to i32
      %2 = arith.addi %1, %arg1 : i32
      tm_tensor.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in region argument types 'i32' and 'i64'}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.extsi %arg1 : i32 to i64
      %2 = arith.addi %1, %arg2 : i64
      tm_tensor.yield %2 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected region to have two arguments}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64, %arg3 : i64):
      %1 = arith.addi %arg1, %arg2 : i64
      tm_tensor.yield %1 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}


// -----

func.func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{mismatch in type of yielded value 'i32' and argument of the region 'i64'}}
      tm_tensor.yield %2 : i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{expected region to yield a single value}}
      tm_tensor.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_index_depth_dynamic(
    %update : tensor<?x?xi64>, %indices : tensor<?x?xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected index depth is static}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x?xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      tm_tensor.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_original_rank_mismatch(
    %update : tensor<?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{op index depth and update value does not cover rank of original value}}
  %0 = tm_tensor.scatter {dimension_map= array<i64: 0>}  unique_indices(true)
    ins(%update, %indices : tensor<?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      tm_tensor.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}
