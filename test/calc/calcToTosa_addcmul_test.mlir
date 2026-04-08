func.func @test_addcmul(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}

func.func @test_addcmul_opt(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}