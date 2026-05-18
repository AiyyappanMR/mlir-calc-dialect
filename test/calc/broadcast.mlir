// RUN: calc-opt --calc-broadcast %s | FileCheck %s

// Test 1: 1D vs 2D — lower rank gets reshaped
// CHECK-LABEL: func.func @test_broadcast_1d_2d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4xf32>, [[ARG1:%.+]]: tensor<1xf32>) -> tensor<1x4xf32>
func.func @test_broadcast_1d_2d_f32(%arg0: tensor<1x4xf32>, %arg1: tensor<1xf32>) -> tensor<1x4xf32> {
    // CHECK-NEXT: [[SHAPE:%.+]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[RESHAPE]] : (tensor<1x4xf32>, tensor<1x1xf32>) -> tensor<1x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<1x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
}
// -----

// Test 2: 1D vs 3D — lower rank gets reshaped
// CHECK-LABEL: func.func @test_broadcast_1d_3d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<2x3x4xf32>
func.func @test_broadcast_1d_3d_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-NEXT: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<4xf32>, !tosa.shape<3>) -> tensor<1x1x4xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[RESHAPE]] : (tensor<2x3x4xf32>, tensor<1x1x4xf32>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<2x3x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}
// -----

// Test 3: 2D vs 3D — lower rank gets reshaped
// CHECK-LABEL: func.func @test_broadcast_2d_3d_i32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xi32>, [[ARG1:%.+]]: tensor<3x4xi32>) -> tensor<2x3x4xi32>
func.func @test_broadcast_2d_3d_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<3x4xi32>) -> tensor<2x3x4xi32> {
    // CHECK-NEXT: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<1x3x4xi32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[RESHAPE]] : (tensor<2x3x4xi32>, tensor<1x3x4xi32>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: return [[RES]] : tensor<2x3x4xi32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<3x4xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
}
// -----

// Test 4: same rank — no reshape should be inserted
// CHECK-LABEL: func.func @test_no_broadcast_same_rank_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<1x4xf32>) -> tensor<4x4xf32>
func.func @test_no_broadcast_same_rank_f32(%arg0: tensor<4x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[ARG1]] : (tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<4x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
}
// -----