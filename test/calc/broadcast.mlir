func.func @test_addcmul_broadcast_2d(%input: tensor<1x4xf32>, %tensor1: tensor<1xf32>) -> tensor<1x4xf32> {
    %result = calc.minimum %input, %tensor1 : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
}