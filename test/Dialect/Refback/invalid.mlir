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
