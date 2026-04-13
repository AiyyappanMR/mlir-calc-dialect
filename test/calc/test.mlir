// RUN: calc-opt --calc-to-arith --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_add_0
// CHECK-SAME: ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @test_add_0(%arg0 : i32, %arg1 : i32)-> i32 {
    // CHECK-NEXT: [[RES:%.+]] = arith.addi [[ARG0]], [[ARG1]] : i32
    // CHECK-NEXT: return [[RES]] : i32
    %result = calc.add %arg0, %arg1 : i32
    return %result : i32
}
// -----

// CHECK-LABEL: func.func @test_add_1
// CHECK-SAME: ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64) -> f64
func.func @test_add_1(%arg0 : f64, %arg1 : f64)-> f64 {
    // CHECK-NEXT: [[RES:%.+]] = arith.addf [[ARG0]], [[ARG1]] : f64
    // CHECK-NEXT: return [[RES]] : f64
    %result = calc.add %arg0, %arg1 : f64
    return %result : f64
}
// -----

// CHECK-LABEL: func.func @test_addt_0
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_addt_0(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT:  return [[RES]] : tensor<4xf32>
    %result = calc.addt %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}
// -----

// CHECK-LABEL: func.func @test_addt_1
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<4xi32>, [[ARG1:%.+]]: tensor<4xi32>) -> tensor<4xi32>
func.func @test_addt_1(%arg0 :tensor<4xi32>,%arg1 : tensor<4xi32>) -> tensor<4xi32>{
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    // CHECK-NEXT:  return [[RES]] : tensor<4xi32>
    %result = calc.addt %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
}
// -----

// CHECK-LABEL: func.func @test_addt_2
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<3x3xi32>, [[ARG1:%.+]]: tensor<3x3xi32>) -> tensor<3x3xi32>
func.func @test_addt_2(%arg0 :tensor<3x3xi32>,%arg1 : tensor<3x3xi32>) -> tensor<3x3xi32>{
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
    // CHECK-NEXT:  return [[RES]] : tensor<3x3xi32>
    %result = calc.addt %arg0, %arg1 : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
    return %result : tensor<3x3xi32>
}
// -----

// CHECK-LABEL: func.func @test_addt_3
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<?xi32>, [[ARG1:%.+]]: tensor<?xi32>) -> tensor<?xi32>
func.func @test_addt_3(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    // CHECK-NEXT:  return [[RES]] : tensor<?xi32>
    %result = calc.addt %arg0, %arg1 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
}
// -----

// CHECK-LABEL: func.func @test_addt_4
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1xi32>, [[ARG1:%.+]]: tensor<?xi32>) -> tensor<1xi32>
func.func @test_addt_4(%arg0 : tensor<1xi32>, %arg1 : tensor<?xi32>) -> tensor<1xi32> {
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<1xi32>, tensor<?xi32>) -> tensor<1xi32>
    // CHECK-NEXT:  return [[RES]] : tensor<1xi32>
    %result = calc.addt %arg0, %arg1 : (tensor<1xi32>, tensor<?xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
}
// -----

// CHECK-LABEL: func.func @test_addt_5
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1xi32>, [[ARG1:%.+]]: tensor<?xi32>) -> tensor<?xi32>
func.func @test_addt_5(%arg0 : tensor<1xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    // CHECK-NEXT:  return [[RES]] : tensor<?xi32>
    %result = calc.addt %arg0, %arg1 : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
}
// -----

// CHECK-LABEL: func.func @test_addt_6
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<3x1xf64>, [[ARG1:%.+]]: tensor<3x3xf64>) -> tensor<3x3xf64>
func.func @test_addt_6(%arg0 :tensor<3x1xf64>,%arg1 : tensor<3x3xf64>) -> tensor<3x3xf64>{
    // CHECK-NEXT:  [[RES:%.+]] = calc.addt [[ARG0]], [[ARG1]] : (tensor<3x1xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    // CHECK-NEXT:  return [[RES]] : tensor<3x3xf64>
    %result = calc.addt %arg0, %arg1 : (tensor<3x1xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %result : tensor<3x3xf64>
}
// -----