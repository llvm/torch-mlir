// RUN: npcomp-opt -split-input-file -verify-diagnostics <%s

// -----

tcp.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{must reference a valid symbol}}
    tcp.get_global_memref @nonexistent_symbol : memref<3xf32>
    return
}

// -----

tcp.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{inconsistent with shape of global}}
    tcp.get_global_memref @g : memref<3xf32>
    return
}

// -----

tcp.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{inconsistent with element type of global}}
    tcp.get_global_memref @g : memref<2xi8>
    return
}

// -----

func @g(%arg0: tensor<?x?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // expected-error @+1 {{number of operands must equal number of results}}
  %add = tcp.shaped_results %arg1, %arg1 {
    %0 = "tcp.add"(%arg0, %arg0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    tcp.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex>, tensor<?xindex> -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}
