// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s

// CHECK-LABEL: @broadcast_to
func @broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // CHECK: tcp.broadcast_to
  %0 = tcp.broadcast_to %arg0, %arg1 : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @splatted
func @splatted(%arg0: f32, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // CHECK: tcp.splatted
  %0 = tcp.splatted %arg0, %arg1 : (f32, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
