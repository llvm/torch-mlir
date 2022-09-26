// RUN: torch-mlir-dialects-opt %s | torch-mlir-dialects-opt | FileCheck %s

// CHECK-LABEL: func.func @test_dummy(%{{.*}}: tensor<10x20xf32>) -> tensor<10x20xf32>
func.func @test_dummy(%arg0 : tensor<10x20xf32>) -> tensor<10x20xf32> {
  // CHECK: %{{.*}} = tcp.dummy %{{.*}} : tensor<10x20xf32> -> tensor<10x20xf32>
  %0 = tcp.dummy %arg0 : tensor<10x20xf32> -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}
