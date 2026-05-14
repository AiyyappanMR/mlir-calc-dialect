// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// Test 1: 1D f32
// CHECK-LABEL: func.func @test_minimum_1d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_minimum_1d_f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[ARG0]], [[ARG1]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[SUB1:%.+]] = tosa.sub [[ARG0]], [[ARG1]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[ABS:%.+]] = tosa.abs [[SUB1]] : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[SUB2:%.+]] = tosa.sub [[ADD]], [[ABS]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[TWO]] : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[SUB2]], [[RECIP]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// Test 2: 2D f32
// CHECK-LABEL: func.func @test_minimum_2d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x3xf32>, [[ARG1:%.+]]: tensor<3x3xf32>) -> tensor<3x3xf32>
func.func @test_minimum_2d_f32(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[ARG0]], [[ARG1]] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-DAG: [[SUB1:%.+]] = tosa.sub [[ARG0]], [[ARG1]] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-DAG: [[ABS:%.+]] = tosa.abs [[SUB1]] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-DAG: [[SUB2:%.+]] = tosa.sub [[ADD]], [[ABS]] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<3x3xf32>}> : () -> tensor<3x3xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[TWO]] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[SUB2]], [[RECIP]], [[SHIFT]] : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi8>) -> tensor<3x3xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<3x3xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
}

// Test 3: 3D f64
// CHECK-LABEL: func.func @test_minimum_3d_f64
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<2x3x4xf64>, [[ARG1:%.+]]: tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
func.func @test_minimum_3d_f64(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[ARG0]], [[ARG1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[SUB1:%.+]] = tosa.sub [[ARG0]], [[ARG1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[ABS:%.+]] = tosa.abs [[SUB1]] : (tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[SUB2:%.+]] = tosa.sub [[ADD]], [[ABS]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<2x3x4xf64>}> : () -> tensor<2x3x4xf64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[TWO]] : (tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[SUB2]], [[RECIP]], [[SHIFT]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>, tensor<1xi8>) -> tensor<2x3x4xf64>
    // CHECK-NEXT: return [[MUL]] : tensor<2x3x4xf64>
    %0 = calc.minimum %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    return %0 : tensor<2x3x4xf64>
}

// Test 4: 1D i32 — integer path uses tosa.int_div instead of reciprocal+mul
// CHECK-LABEL: func.func @test_minimum_1d_i32
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<5xi32>, [[ARG1:%.+]]: tensor<5xi32>) -> tensor<5xi32>
func.func @test_minimum_1d_i32(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<5xi32> {
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[ARG0]], [[ARG1]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    // CHECK-DAG: [[SUB1:%.+]] = tosa.sub [[ARG0]], [[ARG1]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    // CHECK-DAG: [[ABS:%.+]] = tosa.abs [[SUB1]] : (tensor<5xi32>) -> tensor<5xi32>
    // CHECK-DAG: [[SUB2:%.+]] = tosa.sub [[ADD]], [[ABS]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2> : tensor<5xi32>}> : () -> tensor<5xi32>
    // CHECK-DAG: [[DIV:%.+]] = tosa.intdiv [[SUB2]], [[TWO]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    // CHECK-NEXT: return [[DIV]] : tensor<5xi32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    return %0 : tensor<5xi32>
}

// Test 5: 2D i64
// CHECK-LABEL: func.func @test_minimum_2d_i64
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<4x2xi64>, [[ARG1:%.+]]: tensor<4x2xi64>) -> tensor<4x2xi64>
func.func @test_minimum_2d_i64(%arg0: tensor<4x2xi64>, %arg1: tensor<4x2xi64>) -> tensor<4x2xi64> {
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[ARG0]], [[ARG1]] : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-DAG: [[SUB1:%.+]] = tosa.sub [[ARG0]], [[ARG1]] : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-DAG: [[ABS:%.+]] = tosa.abs [[SUB1]] : (tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-DAG: [[SUB2:%.+]] = tosa.sub [[ADD]], [[ABS]] : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
    // CHECK-DAG: [[DIV:%.+]] = tosa.intdiv [[SUB2]], [[TWO]] : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-NEXT: return [[DIV]] : tensor<4x2xi64>
    %0 = calc.minimum %arg0, %arg1 : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    return %0 : tensor<4x2xi64>
}
