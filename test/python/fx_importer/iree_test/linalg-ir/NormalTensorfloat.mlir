#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
    %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
    %cst = arith.constant 5.000000e+00 : f32
    %cst_0 = arith.constant -2.000000e+00 : f32
    %cst_1 = arith.constant 5.42101086E-20 : f32
    %c128_i64 = arith.constant 128 : i64
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c32_i64 = arith.constant 32 : i64
    %cst_3 = arith.constant 6.283180e+00 : f64
    %global_seed = ml_program.global_load @global_seed : tensor<i64>
    %extracted = tensor.extract %global_seed[] : tensor<i64>
    %0 = arith.muli %extracted, %c6364136223846793005_i64 : i64
    %1 = arith.addi %0, %c1442695040888963407_i64 : i64
    %inserted = tensor.insert %1 into %global_seed[] : tensor<i64>
    ml_program.global_store @global_seed = %inserted : tensor<i64>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i64
      %16 = linalg.index 1 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.muli %15, %c128_i64 : i64
      %19 = arith.addi %18, %17 : i64
      %20 = arith.muli %19, %1 : i64
      %21 = arith.addi %20, %1 : i64
      %22 = arith.muli %20, %20 : i64
      %23 = arith.addi %22, %20 : i64
      %24 = arith.shli %23, %c32_i64 : i64
      %25 = arith.shrui %23, %c32_i64 : i64
      %26 = arith.ori %24, %25 : i64
      %27 = arith.muli %26, %26 : i64
      %28 = arith.addi %27, %21 : i64
      %29 = arith.shli %28, %c32_i64 : i64
      %30 = arith.shrui %28, %c32_i64 : i64
      %31 = arith.ori %29, %30 : i64
      %32 = arith.muli %31, %31 : i64
      %33 = arith.addi %32, %20 : i64
      %34 = arith.shli %33, %c32_i64 : i64
      %35 = arith.shrui %33, %c32_i64 : i64
      %36 = arith.ori %34, %35 : i64
      %37 = arith.muli %36, %36 : i64
      %38 = arith.addi %37, %21 : i64
      %39 = arith.shli %38, %c32_i64 : i64
      %40 = arith.shrui %38, %c32_i64 : i64
      %41 = arith.ori %39, %40 : i64
      %42 = arith.muli %41, %41 : i64
      %43 = arith.addi %42, %20 : i64
      %44 = arith.shrui %43, %c32_i64 : i64
      %45 = arith.xori %38, %44 : i64
      %46 = arith.uitofp %45 : i64 to f32
      %47 = arith.mulf %46, %cst_1 : f32
      %48 = arith.addf %47, %cst_2 : f32
      linalg.yield %48 : f32
    } -> tensor<128x128xf32>
    %global_seed_4 = ml_program.global_load @global_seed : tensor<i64>
    %extracted_5 = tensor.extract %global_seed_4[] : tensor<i64>
    %4 = arith.muli %extracted_5, %c6364136223846793005_i64 : i64
    %5 = arith.addi %4, %c1442695040888963407_i64 : i64
    %inserted_6 = tensor.insert %5 into %global_seed_4[] : tensor<i64>
    ml_program.global_store @global_seed = %inserted_6 : tensor<i64>
    %6 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i64
      %16 = linalg.index 1 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.muli %15, %c128_i64 : i64
      %19 = arith.addi %18, %17 : i64
      %20 = arith.muli %19, %5 : i64
      %21 = arith.addi %20, %5 : i64
      %22 = arith.muli %20, %20 : i64
      %23 = arith.addi %22, %20 : i64
      %24 = arith.shli %23, %c32_i64 : i64
      %25 = arith.shrui %23, %c32_i64 : i64
      %26 = arith.ori %24, %25 : i64
      %27 = arith.muli %26, %26 : i64
      %28 = arith.addi %27, %21 : i64
      %29 = arith.shli %28, %c32_i64 : i64
      %30 = arith.shrui %28, %c32_i64 : i64
      %31 = arith.ori %29, %30 : i64
      %32 = arith.muli %31, %31 : i64
      %33 = arith.addi %32, %20 : i64
      %34 = arith.shli %33, %c32_i64 : i64
      %35 = arith.shrui %33, %c32_i64 : i64
      %36 = arith.ori %34, %35 : i64
      %37 = arith.muli %36, %36 : i64
      %38 = arith.addi %37, %21 : i64
      %39 = arith.shli %38, %c32_i64 : i64
      %40 = arith.shrui %38, %c32_i64 : i64
      %41 = arith.ori %39, %40 : i64
      %42 = arith.muli %41, %41 : i64
      %43 = arith.addi %42, %20 : i64
      %44 = arith.shrui %43, %c32_i64 : i64
      %45 = arith.xori %38, %44 : i64
      %46 = arith.uitofp %45 : i64 to f32
      %47 = arith.mulf %46, %cst_1 : f32
      %48 = arith.addf %47, %cst_2 : f32
      linalg.yield %48 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = math.log %in : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.mulf %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = math.sqrt %in : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.truncf %cst_3 : f64 to f32
      %15 = arith.mulf %in, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = math.cos %in : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %11 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %14 = arith.mulf %in, %in_7 : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %cst : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    return %13, %13 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
