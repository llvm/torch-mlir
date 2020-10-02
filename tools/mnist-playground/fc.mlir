
func @fc(
  // TODO: Implement "reshape" so that %image can be passed as batch of 2D tensors.
  %image: tensor<?x?xf32>,
  %weights: tensor<?x?xf32>,
  %biases: tensor<?x?xf32>)
-> (
  tensor<?x?xf32>
) {
  %0 = tcf.matmul %weights, %image : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tcf.add %0, %biases : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // TODO: Implement softmax for classification.
  // For now, this returns a not-terribly useful number.
  return %1 : tensor<?x?xf32>
}
