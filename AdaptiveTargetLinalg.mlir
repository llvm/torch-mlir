func.func @forward(%arg0: tensor<?x?xf32>) -> tensor<?x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // number of batches*channels
    %N = tensor.dim %arg0, %c0 : tensor<?x?xf32> 
    // length of input tensors
    %lin = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    //length of output tensors
    %lout = arith.constant 7 : index 
    %Kmax = arith.ceildivsi %lin, %lout : index 
    //an empty tensor to use for iterating over each window (maximum size = lin)
    %Kiter = tensor.empty(%Kmax) : tensor<?xi1> 
    // 0 or -inf dep. on whether doing avg or max, resp.
    %buffval = arith.constant 0.0 : f32 
    //buffering the input since otherwise the output loop will try to access %arg0 out of bounds
    %buffdinput = tensor.pad %arg0 low[0,0] high[0,1] {
        ^bb0(%arg7: index, %arg8 : index):
            tensor.yield %buffval : f32
        } : tensor<?x?xf32> to tensor<?x?xf32> 
    //start index for each window: start_index(i) = (i/lout)*lin + ((i%lout)*lin)/lout = (i*lin)/lout
    %st_init = tensor.empty(%lout) : tensor<?xindex> 
    //end index for each window: end_index(i) = 1 + ((i + 1)*lin - 1)/lout
    %en_init = tensor.empty(%lout) : tensor<?xindex> 
    %st , %en = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], 
        iterator_types = ["parallel"]} outs(%st_init, %en_init: tensor<? x index>, tensor<?xindex>) {
            ^bb0(%arg1 : index, %arg2 : index):
                %l = linalg.index 0 : index
                %s3 = arith.muli %l, %lin : index
                %s4 = arith.floordivsi %s3, %lout : index
                %e2 = arith.addi %l, %c1 : index
                %e3 = arith.muli %e2, %lin : index
                %e4 = arith.subi %e3, %c1 : index
                %e5 = arith.floordivsi %e4, %lout : index
                %e6 = arith.addi %c1, %e5 : index
                linalg.yield %s4, %e6 : index, index
    } -> (tensor<?xindex>, tensor<? x index>)
    %init = tensor.empty(%lin) : tensor<?x7xf32> 
    //pool op result over windows [st,en] in arg0 (in this case, does a sum pool)
    %output = linalg.generic{
        indexing_maps = [affine_map<(d0,d1,d2) -> (d1)>, affine_map<(d0,d1,d2) -> (d1)>, affine_map<(d0,d1,d2) -> (d2)>, affine_map<(d0,d1,d2) -> (d0,d1)>],
        iterator_types = ["parallel","parallel","reduction"]} ins(%st, %en, %Kiter: tensor<?xindex>, tensor<?xindex>, tensor<?xi1>) outs(%init: tensor<?x7xf32>) {
            ^bb0(%arg3 : index, %arg4 : index, %arg5 : i8, %arg6 : f32):
                %ind0 = linalg.index 0 : index
                %ind2 = linalg.index 2 : index
                %sum = arith.addi %arg3, %ind2 : index
                %inelt = tensor.extract %buffdinput [%ind0, %sum] : tensor<?x?xf32>
                %cond = arith.cmpi ult, %ind2, %arg4 : index 
                %outelt1 = arith.select %cond, %inelt, %buffval : f32
                //replace with maxf for adaptive max
                %outelt2 = arith.addf %arg6, %outelt1 : f32 
                linalg.yield %outelt2 : f32
    } -> tensor<?x7xf32>
    //for avg, need to add loop to float-divide %output, elementwise in dim 1, by (%en - %st).
    return %output : tensor<?x7xf32>
}