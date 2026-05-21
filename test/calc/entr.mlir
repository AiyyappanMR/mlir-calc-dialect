// This file contains MLIR IR for testing the lowering of calc.entr to tosa dialect.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// Test 1: 1D f32
// CHECK-LABEL: func.func @test_entr_1d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_entr_1d_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    // CHECK-DAG: [[EQMASK:%.+]] = tosa.equal [[ARG0]], [[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    // CHECK-DAG: [[LTMASK:%.+]] = tosa.greater [[ZERO]], [[ARG0]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[LOG:%.+]] = tosa.log [[ARG0]] : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[ARG0]], [[LOG]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-DAG: [[NEG:%.+]] = tosa.negate [[MUL]], {{%.+}}, {{%.+}} : (tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[NEGINF:%.+]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<4xf32>}> : () -> tensor<4xf32>
    // CHECK-DAG: [[OUT1:%.+]] = tosa.select [[EQMASK]], [[ZERO]], [[NEG]] : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[OUT2:%.+]] = tosa.select [[LTMASK]], [[NEGINF]], [[OUT1]] : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: return [[OUT2]] : tensor<4xf32>
    %0 = calc.entr %arg0 : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// Test 2: 2D f64
// CHECK-LABEL: func.func @test_entr_2d_f64
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x3xf64>) -> tensor<3x3xf64>
func.func @test_entr_2d_f64(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<3x3xf64>}> : () -> tensor<3x3xf64>
    // CHECK-DAG: [[EQMASK:%.+]] = tosa.equal [[ARG0]], [[ZERO]] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    // CHECK-DAG: [[LTMASK:%.+]] = tosa.greater [[ZERO]], [[ARG0]] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[LOG:%.+]] = tosa.log [[ARG0]] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[ARG0]], [[LOG]], [[SHIFT]] : (tensor<3x3xf64>, tensor<3x3xf64>, tensor<1xi8>) -> tensor<3x3xf64>
    // CHECK-DAG: [[NEG:%.+]] = tosa.negate [[MUL]], {{%.+}}, {{%.+}} : (tensor<3x3xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3x3xf64>
    // CHECK-DAG: [[NEGINF:%.+]] = "tosa.const"() <{values = dense<0xFFF0000000000000> : tensor<3x3xf64>}> : () -> tensor<3x3xf64>
    // CHECK-DAG: [[OUT1:%.+]] = tosa.select [[EQMASK]], [[ZERO]], [[NEG]] : (tensor<3x3xi1>, tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    // CHECK-DAG: [[OUT2:%.+]] = tosa.select [[LTMASK]], [[NEGINF]], [[OUT1]] : (tensor<3x3xi1>, tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    // CHECK-NEXT: return [[OUT2]] : tensor<3x3xf64>
    %0 = calc.entr %arg0 : (tensor<3x3xf64>) -> tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
}

// Test 3: 2D i32 -> f32
// CHECK-LABEL: func.func @test_entr_1d_i32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4xi32>) -> tensor<1x4xf32>
func.func @test_entr_1d_i32(%arg0: tensor<1x4xi32>) -> tensor<1x4xf32> {
    // CHECK-DAG: [[CAST:%.+]] = tosa.cast [[ARG0]] : (tensor<1x4xi32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x4xf32>}> : () -> tensor<1x4xf32>
    // CHECK-DAG: [[EQMASK:%.+]] = tosa.equal [[CAST]], [[ZERO]] : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    // CHECK-DAG: [[LTMASK:%.+]] = tosa.greater [[ZERO]], [[CAST]] : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[LOG:%.+]] = tosa.log [[CAST]] : (tensor<1x4xf32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAST]], [[LOG]], [[SHIFT]] : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1xi8>) -> tensor<1x4xf32>
    // CHECK-DAG: [[NEG:%.+]] = tosa.negate [[MUL]], {{%.+}}, {{%.+}} : (tensor<1x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[NEGINF:%.+]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4xf32>}> : () -> tensor<1x4xf32>
    // CHECK-DAG: [[OUT1:%.+]] = tosa.select [[EQMASK]], [[ZERO]], [[NEG]] : (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[OUT2:%.+]] = tosa.select [[LTMASK]], [[NEGINF]], [[OUT1]] : (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    // CHECK-NEXT: return [[OUT2]] : tensor<1x4xf32>
    %0 = calc.entr %arg0 : (tensor<1x4xi32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
}

// Test 4: 3D i64 -> f64
// CHECK-LABEL: func.func @test_entr_1d_i64
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xi64>) -> tensor<2x3x4xf64>
func.func @test_entr_1d_i64(%arg0: tensor<2x3x4xi64>) -> tensor<2x3x4xf64> {
    // CHECK-DAG: [[CAST:%.+]] = tosa.cast [[ARG0]] : (tensor<2x3x4xi64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x3x4xf64>}> : () -> tensor<2x3x4xf64>
    // CHECK-DAG: [[EQMASK:%.+]] = tosa.equal [[CAST]], [[ZERO]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xi1>
    // CHECK-DAG: [[LTMASK:%.+]] = tosa.greater [[ZERO]], [[CAST]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xi1>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[LOG:%.+]] = tosa.log [[CAST]] : (tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAST]], [[LOG]], [[SHIFT]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>, tensor<1xi8>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[NEG:%.+]] = tosa.negate [[MUL]], {{%.+}}, {{%.+}} : (tensor<2x3x4xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[NEGINF:%.+]] = "tosa.const"() <{values = dense<0xFFF0000000000000> : tensor<2x3x4xf64>}> : () -> tensor<2x3x4xf64>
    // CHECK-DAG: [[OUT1:%.+]] = tosa.select [[EQMASK]], [[ZERO]], [[NEG]] : (tensor<2x3x4xi1>, tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-DAG: [[OUT2:%.+]] = tosa.select [[LTMASK]], [[NEGINF]], [[OUT1]] : (tensor<2x3x4xi1>, tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
    // CHECK-NEXT: return [[OUT2]] : tensor<2x3x4xf64>
    %0 = calc.entr %arg0 : (tensor<2x3x4xi64>) -> tensor<2x3x4xf64>
    return %0 : tensor<2x3x4xf64>
}