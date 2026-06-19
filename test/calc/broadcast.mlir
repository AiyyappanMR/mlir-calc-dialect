// Tests broadcasting for calc dialect operations using calc-broadcast pass.

// RUN: calc-opt --calc-broadcast %s | FileCheck %s

// Test 1: 1D vs 2D — lower rank gets reshaped, then tiled to match shape
// CHECK-LABEL: func.func @test_broadcast_1d_2d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4xf32>, [[ARG1:%.+]]: tensor<1xf32>) -> tensor<1x4xf32>
func.func @test_broadcast_1d_2d_f32(%arg0: tensor<1x4xf32>, %arg1: tensor<1xf32>) -> tensor<1x4xf32> {
    // CHECK-NEXT: [[T_SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: [[R_SHAPE:%.+]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[R_SHAPE]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    // CHECK-NEXT: [[TILED:%.+]] = tosa.tile [[RESHAPE]], [[T_SHAPE]] : (tensor<1x1xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[TILED]] : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<1x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
}
// -----

// Test 2: 1D vs 3D — lower rank gets reshaped, then tiled across upper broadcast dimensions
// CHECK-LABEL: func.func @test_broadcast_1d_3d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<2x3x4xf32>
func.func @test_broadcast_1d_3d_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-NEXT: [[T_SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[R_SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[R_SHAPE]] : (tensor<4xf32>, !tosa.shape<3>) -> tensor<1x1x4xf32>
    // CHECK-NEXT: [[TILED:%.+]] = tosa.tile [[RESHAPE]], [[T_SHAPE]] : (tensor<1x1x4xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[TILED]] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<2x3x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}
// -----

// Test 3: 2D vs 3D — lower rank gets reshaped, then tiled on leading unit dimension
// CHECK-LABEL: func.func @test_broadcast_2d_3d_i32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xi32>, [[ARG1:%.+]]: tensor<3x4xi32>) -> tensor<2x3x4xi32>
func.func @test_broadcast_2d_3d_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<3x4xi32>) -> tensor<2x3x4xi32> {
    // CHECK-NEXT: [[T_SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[R_SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[RESHAPE:%.+]] = tosa.reshape [[ARG1]], [[R_SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<1x3x4xi32>
    // CHECK-NEXT: [[TILED:%.+]] = tosa.tile [[RESHAPE]], [[T_SHAPE]] : (tensor<1x3x4xi32>, !tosa.shape<3>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[TILED]] : (tensor<2x3x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: return [[RES]] : tensor<2x3x4xi32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<3x4xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
}
// -----

// Test 4: Same rank with different shapes — no rank reshape, but triggers shape equalization tiling
// CHECK-LABEL: func.func @test_no_broadcast_same_rank_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<1x4xf32>) -> tensor<4x4xf32>
func.func @test_no_broadcast_same_rank_f32(%arg0: tensor<4x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: [[T_SHAPE:%.+]] = tosa.const_shape {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: [[TILED:%.+]] = tosa.tile [[ARG1]], [[T_SHAPE]] : (tensor<1x4xf32>, !tosa.shape<2>) -> tensor<4x4xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[ARG0]], [[TILED]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<4x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
}
// -----

// Test 5: Mixed rank and independent cross-tiling requirements
// CHECK-LABEL: func.func @test_no_broadcast_shape
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<4x1x4xf32>) -> tensor<4x4x4xf32>
func.func @test_no_broadcast_shape(%arg0: tensor<4x4xf32>, %arg1: tensor<4x1x4xf32>) -> tensor<4x4x4xf32> {
    // CHECK-NEXT: [[T_SHAPE1:%.+]] = tosa.const_shape {values = dense<[1, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[T_SHAPE0:%.+]] = tosa.const_shape {values = dense<[4, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[R_SHAPE0:%.+]] = tosa.const_shape {values = dense<[1, 4, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: [[RESHAPE0:%.+]] = tosa.reshape [[ARG0]], [[R_SHAPE0]] : (tensor<4x4xf32>, !tosa.shape<3>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: [[TILED0:%.+]] = tosa.tile [[RESHAPE0]], [[T_SHAPE0]] : (tensor<1x4x4xf32>, !tosa.shape<3>) -> tensor<4x4x4xf32>
    // CHECK-NEXT: [[TILED1:%.+]] = tosa.tile [[ARG1]], [[T_SHAPE1]] : (tensor<4x1x4xf32>, !tosa.shape<3>) -> tensor<4x4x4xf32>
    // CHECK-NEXT: [[RES:%.+]] = calc.minimum [[TILED0]], [[TILED1]] : (tensor<4x4x4xf32>, tensor<4x4x4xf32>) -> tensor<4x4x4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<4x4x4xf32>
    %0 = calc.minimum %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x1x4xf32>) -> tensor<4x4x4xf32>
    return %0 : tensor<4x4x4xf32>
}