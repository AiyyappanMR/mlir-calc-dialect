// Tests lowering of calc dialect basic operations to tosa dialect.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @test_add_0
// CHECK-SAME:  ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @test_add_0(%arg0 : i32, %arg1 : i32)-> i32 {
    // CHECK:      [[C0:%.+]] = tensor.from_elements [[ARG0]] : tensor<i32>
    // CHECK-NEXT: [[C1:%.+]] = tensor.from_elements [[ARG1]] : tensor<i32>
    // CHECK-NEXT: [[TEN_RES:%.+]] = tosa.add [[C0]], [[C1]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK-NEXT: [[SCA_RES:%.+]] = tensor.extract [[TEN_RES]][] : tensor<i32>
    // CHECK-NEXT: return [[SCA_RES]] : i32
    %result = calc.add %arg0, %arg1 : i32
    return %result : i32
}
// -----

// CHECK-LABEL: func.func @test_add_1
// CHECK-SAME:  ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64) -> f64
func.func @test_add_1(%arg0 : f64, %arg1 : f64)-> f64 {
    // CHECK:      [[C0:%.+]] = tensor.from_elements [[ARG0]] : tensor<f64>
    // CHECK-NEXT: [[C1:%.+]] = tensor.from_elements [[ARG1]] : tensor<f64>
    // CHECK-NEXT: [[TEN_RES:%.+]] = tosa.add [[C0]], [[C1]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
    // CHECK-NEXT: [[SCA_RES:%.+]] = tensor.extract [[TEN_RES]][] : tensor<f64>
    // CHECK-NEXT: return [[SCA_RES]] : f64
    %result = calc.add %arg0, %arg1 : f64
    return %result : f64
}
// -----

// CHECK-LABEL: func.func @test_mul_0
// CHECK-SAME:  ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @test_mul_0(%arg0 : i32, %arg1 : i32)-> i32 {
    // CHECK:      [[C0:%.+]] = tensor.from_elements [[ARG0]] : tensor<i32>
    // CHECK-NEXT: [[C1:%.+]] = tensor.from_elements [[ARG1]] : tensor<i32>
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<i8>}> : () -> tensor<*xi8>
    // CHECK-NEXT: [[TEN_RES:%.+]] = tosa.mul [[C0]], [[C1]], [[SHIFT]] : (tensor<i32>, tensor<i32>, tensor<*xi8>) -> tensor<i32>
    // CHECK-NEXT: [[SCA_RES:%.+]] = tensor.extract [[TEN_RES]][] : tensor<i32>
    // CHECK-NEXT: return [[SCA_RES]] : i32
    %result = calc.mul %arg0, %arg1 : i32
    return %result : i32
}
// -----

// CHECK-LABEL: func.func @test_mul_1
// CHECK-SAME:  ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64) -> f64
func.func @test_mul_1(%arg0 : f64, %arg1 : f64)-> f64 {
    // CHECK:      [[C0:%.+]] = tensor.from_elements [[ARG0]] : tensor<f64>
    // CHECK-NEXT: [[C1:%.+]] = tensor.from_elements [[ARG1]] : tensor<f64>
    // CHECK-NEXT: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<i8>}> : () -> tensor<*xi8>
    // CHECK-NEXT: [[TEN_RES:%.+]] = tosa.mul [[C0]], [[C1]], [[SHIFT]] : (tensor<f64>, tensor<f64>, tensor<*xi8>) -> tensor<f64>
    // CHECK-NEXT: [[SCA_RES:%.+]] = tensor.extract [[TEN_RES]][] : tensor<f64>
    // CHECK-NEXT: return [[SCA_RES]] : f64
    %result = calc.mul %arg0, %arg1 : f64
    return %result : f64
}
// -----

