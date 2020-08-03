// RUN: npcomp-opt -resolve-tensor-load-store-ops <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @basic
func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  %shape = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> tensor<?xindex>

  // CHECK: %[[SRCMEMREF:.+]] = tcp.alloc_memref
  %src_memref = tcp.alloc_memref %shape : memref<?xf32>
  // tensor_store of argument remains.
  // CHECK: tensor_store %arg0, %[[SRCMEMREF]]
  tensor_store %arg0, %src_memref : memref<?xf32>
  %src = tensor_load %src_memref : memref<?xf32>

  // CHECK: %[[DSTMEMREF:.+]] = tcp.alloc_memref
  %dst_memref = tcp.alloc_memref %shape : memref<?xf32>
  // tensor_store of internally created tensor is eliminated.
  // CHECK-NOT: tensor_store
  // CHECK: linalg.copy(%[[SRCMEMREF]], %[[DSTMEMREF]])
  tensor_store %src, %dst_memref : memref<?xf32>
  %ret = tensor_load %dst_memref : memref<?xf32>

  // The tensor_load feeding into the return remains.
  // %[[RET:.+]] = tensor_load %[[DSTMEMREF]]
  // return %[[RET]]
  return %ret : tensor<?xf32>
}
