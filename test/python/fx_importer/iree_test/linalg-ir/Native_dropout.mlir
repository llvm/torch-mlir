#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<f64>
    %c32_i64 = arith.constant 32 : i64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c128_i64 = arith.constant 128 : i64
    %cst_1 = arith.constant 5.4210107999999998E-20 : f64
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
    %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
    %global_seed = ml_program.global_load @global_seed : tensor<i64>
    %extracted = tensor.extract %global_seed[] : tensor<i64>
    %0 = arith.muli %extracted, %c6364136223846793005_i64 : i64
    %1 = arith.addi %0, %c1442695040888963407_i64 : i64
    %inserted = tensor.insert %1 into %global_seed[] : tensor<i64>
    ml_program.global_store @global_seed = %inserted : tensor<i64>
    %2 = tensor.empty() : tensor<128x128xf64>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x128xf64>) {
    ^bb0(%out: f64):
      %10 = linalg.index 0 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.muli %11, %c128_i64 : i64
      %15 = arith.addi %14, %13 : i64
      %16 = arith.muli %15, %1 : i64
      %17 = arith.addi %16, %1 : i64
      %18 = arith.muli %16, %16 : i64
      %19 = arith.addi %18, %16 : i64
      %20 = arith.shli %19, %c32_i64 : i64
      %21 = arith.shrui %19, %c32_i64 : i64
      %22 = arith.ori %20, %21 : i64
      %23 = arith.muli %22, %22 : i64
      %24 = arith.addi %23, %17 : i64
      %25 = arith.shli %24, %c32_i64 : i64
      %26 = arith.shrui %24, %c32_i64 : i64
      %27 = arith.ori %25, %26 : i64
      %28 = arith.muli %27, %27 : i64
      %29 = arith.addi %28, %16 : i64
      %30 = arith.shli %29, %c32_i64 : i64
      %31 = arith.shrui %29, %c32_i64 : i64
      %32 = arith.ori %30, %31 : i64
      %33 = arith.muli %32, %32 : i64
      %34 = arith.addi %33, %17 : i64
      %35 = arith.shli %34, %c32_i64 : i64
      %36 = arith.shrui %34, %c32_i64 : i64
      %37 = arith.ori %35, %36 : i64
      %38 = arith.muli %37, %37 : i64
      %39 = arith.addi %38, %16 : i64
      %40 = arith.shrui %39, %c32_i64 : i64
      %41 = arith.xori %34, %40 : i64
      %42 = arith.uitofp %41 : i64 to f64
      %43 = arith.mulf %42, %cst_1 : f64
      %44 = arith.addf %43, %cst_0 : f64
      linalg.yield %44 : f64
    } -> tensor<128x128xf64>
    %4 = tensor.empty() : tensor<128x128xi1>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst : tensor<128x128xf64>, tensor<f64>) outs(%4 : tensor<128x128xi1>) {
    ^bb0(%in: f64, %in_3: f64, %out: i1):
      %10 = arith.cmpf olt, %in, %in_3 : f64
      linalg.yield %10 : i1
    } -> tensor<128x128xi1>
    %6 = tensor.empty() : tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<128x128xi1>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %out: f32):
      %10 = arith.uitofp %in : i1 to f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %10 = arith.mulf %in, %in_3 : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<128x128xf32>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.divf %in, %cst_2 : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    return %9 : tensor<128x128xf32>
  }
}
