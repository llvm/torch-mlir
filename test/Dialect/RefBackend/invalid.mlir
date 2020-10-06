// RUN: npcomp-opt -split-input-file -verify-diagnostics <%s

// -----

refback.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{must reference a valid symbol}}
    refback.get_global_memref @nonexistent_symbol : memref<3xf32>
    return
}

// -----

refback.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{inconsistent with shape of global}}
    refback.get_global_memref @g : memref<3xf32>
    return
}

// -----

refback.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{inconsistent with element type of global}}
    refback.get_global_memref @g : memref<2xi8>
    return
}

// -----

func @g(%arg0: tensor<?x?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // expected-error @+1 {{number of operands must equal number of results}}
  %add = refback.shaped_results %arg1, %arg1 {
    %0 = tcp.add %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    refback.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex>, tensor<?xindex> -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}
