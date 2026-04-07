// Test 1 — f32
func.func @test_addcmul_f32(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}

// Test 2 — f64
func.func @test_addcmul_f64(%input: tensor<4xf64>, %tensor1: tensor<4xf64>, %tensor2: tensor<4xf64>) -> tensor<4xf64> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1 : i64} : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %result : tensor<4xf64>
}

// Test 3 — i32
func.func @test_addcmul_i32(%input: tensor<4xi32>, %tensor1: tensor<4xi32>, %tensor2: tensor<4xi32>) -> tensor<4xi32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 5.0 : f64} : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
}

// Test 4 — i64
func.func @test_addcmul_i64(%input: tensor<4xi64>, %tensor1: tensor<4xi64>, %tensor2: tensor<4xi64>) -> tensor<4xi64> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 2 : i64} : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
}

// Test 5 — 2D tensors
func.func @test_addcmul_2d(%input: tensor<2x4xf32>, %tensor1: tensor<2x4xf32>, %tensor2: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
}

// Test 6 — 3D tensors
func.func @test_addcmul_3d(%input: tensor<2x3x4xf32>, %tensor1: tensor<2x3x4xf32>, %tensor2: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
}

// Test 7 — broadcasting 1D
func.func @test_addcmul_broadcast(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<1xf32>) -> tensor<4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}

// Test 8 — broadcasting 2D
func.func @test_addcmul_broadcast_2d(%input: tensor<2x4xf32>, %tensor1: tensor<2x4xf32>, %tensor2: tensor<1x4xf32>) -> tensor<2x4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
}

// Test 9 — dynamic 1D
func.func @test_addcmul_dynamic(%input: tensor<?xf32>, %tensor1: tensor<?xf32>, %tensor2: tensor<?xf32>) -> tensor<?xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %result : tensor<?xf32>
}

// Test 10 — dynamic 2D
func.func @test_addcmul_dynamic_2d(%input: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
}

// Test 11- optionally constant value
func.func @test_addcmul_opt(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    %result = calc.addcmul %input, %tensor1, %tensor2 : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}