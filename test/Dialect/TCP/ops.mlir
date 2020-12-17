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

// CHECK-LABEL: @pad
func @pad(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xindex>, %arg2: tensor<?xindex>, %arg3: f32) -> tensor<?x?x?x?xf32> {
  // CHECK: tcp.pad
  %0 = tcp.pad %arg0, %arg1, %arg2, %arg3 : (tensor<?x?x?x?xf32>, tensor<?xindex>, tensor<?xindex>, f32) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
