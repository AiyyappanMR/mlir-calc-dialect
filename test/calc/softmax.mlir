// RUN: calc-opt %s --calc-to-tosa | FileCheck %s

// Test 1: 2D f32 dim=0
// CHECK-LABEL: func.func @softmax_2d_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @softmax_2d_dim0(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 0 : i32} : (tensor<4x4xf32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<1x4xf32>) -> tensor<1x4xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<4x4xf32>, tensor<1x4xf32>, tensor<1xi8>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<4x4xf32>
    %result = calc.softmax %input {dim = 0 : si32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}

// Test 2: 2D f32 dim=1
// CHECK-LABEL: func.func @softmax_2d_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @softmax_2d_dim1(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 1 : i32} : (tensor<4x4xf32>) -> tensor<4x1xf32>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<4x1xf32>) -> tensor<4x1xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<4x4xf32>, tensor<4x1xf32>, tensor<1xi8>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<4x4xf32>
    %result = calc.softmax %input {dim = 1 : si32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}

// Test 3: 2D f32 neg dim=-1
// CHECK-LABEL: func.func @softmax_neg_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @softmax_neg_dim1(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 1 : i32} : (tensor<4x4xf32>) -> tensor<4x1xf32>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<4x1xf32>) -> tensor<4x1xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<4x4xf32>, tensor<4x1xf32>, tensor<1xi8>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<4x4xf32>
    %result = calc.softmax %input {dim = -1 : si32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}

// Test 4: 3D f32 neg dim=-2
// CHECK-LABEL: func.func @softmax_3d_neg_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
func.func @softmax_3d_neg_dim2(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 1 : i32} : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<2x4x8xf32>, tensor<2x1x8xf32>, tensor<1xi8>) -> tensor<2x4x8xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<2x4x8xf32>
    %result = calc.softmax %input {dim = -2 : si32} : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %result : tensor<2x4x8xf32>
}

// Test 5: 1D f32 dim=0
// CHECK-LABEL: func.func @softmax_1d
// CHECK-SAME: ([[ARG0:%.+]]: tensor<8xf32>) -> tensor<8xf32>
func.func @softmax_1d(%input: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 0 : i32} : (tensor<8xf32>) -> tensor<1xf32>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<1xf32>) -> tensor<1xf32>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<8xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<8xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<8xf32>
    %result = calc.softmax %input {dim = 0 : si32} : (tensor<8xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
}

// Test 6: 3D f64 dim=2
// CHECK-LABEL: func.func @softmax_3d_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4x8xf64>) -> tensor<2x4x8xf64>
func.func @softmax_3d_dim2(%input: tensor<2x4x8xf64>) -> tensor<2x4x8xf64> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<2x4x8xf64>) -> tensor<2x4x8xf64>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 2 : i32} : (tensor<2x4x8xf64>) -> tensor<2x4x1xf64>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<2x4x1xf64>) -> tensor<2x4x1xf64>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<2x4x8xf64>, tensor<2x4x1xf64>, tensor<1xi8>) -> tensor<2x4x8xf64>
    // CHECK-NEXT: return [[MUL]] : tensor<2x4x8xf64>
    %result = calc.softmax %input {dim = 2 : si32} : (tensor<2x4x8xf64>) -> tensor<2x4x8xf64>
    return %result : tensor<2x4x8xf64>
}

// Test 7: 4D f64 dim=3
// CHECK-LABEL: func.func @softmax_4d
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x8x16x16xf64>) -> tensor<2x8x16x16xf64>
func.func @softmax_4d(%input: tensor<2x8x16x16xf64>) -> tensor<2x8x16x16xf64> {
    // CHECK-DAG: [[EXP:%.+]] = tosa.exp [[ARG0]] : (tensor<2x8x16x16xf64>) -> tensor<2x8x16x16xf64>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_sum [[EXP]] {axis = 3 : i32} : (tensor<2x8x16x16xf64>) -> tensor<2x8x16x1xf64>
    // CHECK-DAG: [[RECIP:%.+]] = tosa.reciprocal [[REDUCE]] : (tensor<2x8x16x1xf64>) -> tensor<2x8x16x1xf64>
    // CHECK-DAG: [[ZERO:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[EXP]], [[RECIP]], [[ZERO]] : (tensor<2x8x16x16xf64>, tensor<2x8x16x1xf64>, tensor<1xi8>) -> tensor<2x8x16x16xf64>
    // CHECK-NEXT: return [[MUL]] : tensor<2x8x16x16xf64>
    %result = calc.softmax %input {dim = 3 : si32} : (tensor<2x8x16x16xf64>) -> tensor<2x8x16x16xf64>
    return %result : tensor<2x8x16x16xf64>
}