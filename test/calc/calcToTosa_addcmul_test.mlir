// Tests lowering of calc.addcmul to tosa dialect with various types.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @test_addcmul
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[TENSOR1:%.+]]: tensor<4xf32>, [[TENSOR2:%.+]]: tensor<4xf32>)
func.func @test_addcmul(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
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

// CHECK-LABEL: func.func @test_addcmul_opt
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[TENSOR1:%.+]]: tensor<4xf32>, [[TENSOR2:%.+]]: tensor<4xf32>)
func.func @test_addcmul_opt(%input: tensor<4xf32>, %tensor1: tensor<4xf32>, %tensor2: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: [[MUL1:%.+]] = tosa.mul [[TENSOR1]], [[TENSOR2]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: [[ADD:%.+]] = tosa.add [[INPUT]], [[MUL1]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: return [[ADD]] : tensor<4xf32>
    %result = calc.addcmul %input, %tensor1, %tensor2 : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}