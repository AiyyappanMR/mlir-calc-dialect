// Tests lowering of calc.addcmul to tosa dialect.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @test_addcmul_f32
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[TENSOR1:%.+]]: tensor<4xf32>, [[TENSOR2:%.+]]: tensor<4xf32>)
func.func @test_addcmul_f32(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_f64
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf64>, [[TENSOR1:%.+]]: tensor<4xf64>, [[TENSOR2:%.+]]: tensor<4xf64>)
func.func @test_addcmul_f64(%input: tensor<4xf64>, %tensor1: tensor<4xf64>, %tensor2: tensor<4xf64>) -> tensor<4xf64> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xf64>, tensor<4xf64>, tensor<1xi8>) -> tensor<4xf64>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4xf64>}> : () -> tensor<4xf64>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<4xf64>, tensor<4xf64>, tensor<1xi8>) -> tensor<4xf64>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    // CHECK-NEXT: return [[ADD]] : tensor<4xf64>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f64} : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %result : tensor<4xf64>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_i32
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xi32>, [[TENSOR1:%.+]]: tensor<4xi32>, [[TENSOR2:%.+]]: tensor<4xi32>)
func.func @test_addcmul_i32(%input: tensor<4xi32>, %tensor1: tensor<4xi32>, %tensor2: tensor<4xi32>) -> tensor<4xi32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<5> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    // CHECK-NEXT: return [[ADD]] : tensor<4xi32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 5 : i32} : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_i64
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xi64>, [[TENSOR1:%.+]]: tensor<4xi64>, [[TENSOR2:%.+]]: tensor<4xi64>)
func.func @test_addcmul_i64(%input: tensor<4xi64>, %tensor1: tensor<4xi64>, %tensor2: tensor<4xi64>) -> tensor<4xi64> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xi64>, tensor<4xi64>, tensor<1xi8>) -> tensor<4xi64>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<2> : tensor<4xi64>}> : () -> tensor<4xi64>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<4xi64>, tensor<4xi64>, tensor<1xi8>) -> tensor<4xi64>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    // CHECK-NEXT: return [[ADD]] : tensor<4xi64>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 2 : i64} : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_2d
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x4xf32>, [[TENSOR1:%.+]]: tensor<2x4xf32>, [[TENSOR2:%.+]]: tensor<2x4xf32>)
func.func @test_addcmul_2d(%input: tensor<2x4xf32>, %tensor1: tensor<2x4xf32>, %tensor2: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x4xf32>}> : () -> tensor<2x4xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<2x4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_3d
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x3x4xf32>, [[TENSOR1:%.+]]: tensor<2x3x4xf32>, [[TENSOR2:%.+]]: tensor<2x3x4xf32>)
func.func @test_addcmul_3d(%input: tensor<2x3x4xf32>, %tensor1: tensor<2x3x4xf32>, %tensor2: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x3x4xf32>}> : () -> tensor<2x3x4xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<2x3x4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_broadcast
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[TENSOR1:%.+]]: tensor<4xf32>, [[TENSOR2:%.+]]: tensor<1xf32>)
func.func @test_addcmul_broadcast(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<1xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}
// -----

// CHECK-LABEL: func.func @test_addcmul_broadcast_2d
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x4xf32>, [[TENSOR1:%.+]]: tensor<2x4xf32>, [[TENSOR2:%.+]]: tensor<1x4xf32>)
func.func @test_addcmul_broadcast_2d(%input: tensor<2x4xf32>, %tensor1: tensor<2x4xf32>, %tensor2: tensor<1x4xf32>) -> tensor<2x4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<2x4xf32>, tensor<1x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
    // CHECK-NEXT: [[VAL:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x4xf32>}> : () -> tensor<2x4xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = tosa.mul [[MUL1]], [[VAL]], [[SHIFT]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL2]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<2x4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 {value = 1.0 : f32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
}