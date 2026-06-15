// RUN: calc-opt %s --calc-to-tosa | FileCheck %s

// 1D inputs

// Test 1: 1D f32, equal split, dim=0
// CHECK-LABEL: func.func @split_1d_f32_equal_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
func.func @split_1d_f32_equal_dim0(%a: tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<4xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xf32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<4xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xf32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 2>} : (tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// Test 2: 1D i32, unequal split, dim=0
// CHECK-LABEL: func.func @split_1d_i32_unequal_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<5xi32>) -> (tensor<1xi32>, tensor<4xi32>)
func.func @split_1d_i32_unequal_dim0(%a: tensor<5xi32>) -> (tensor<1xi32>, tensor<4xi32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<5xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xi32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<5xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 1, 4>} : (tensor<5xi32>) -> (tensor<1xi32>, tensor<4xi32>)
    return %0, %1 : tensor<1xi32>, tensor<4xi32>
}

// Test 3: 1D f64, 3-way split, dim=0
// CHECK-LABEL: func.func @split_1d_f64_3way_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<6xf64>) -> (tensor<1xf64>, tensor<2xf64>, tensor<3xf64>)
func.func @split_1d_f64_3way_dim0(%a: tensor<6xf64>) -> (tensor<1xf64>, tensor<2xf64>, tensor<3xf64>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<6xf64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xf64>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<6xf64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xf64>
    // CHECK-DAG: [[START2:%.+]] = tosa.const_shape  {values = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE2:%.+]] = tosa.const_shape  {values = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE2:%.+]] = tosa.slice [[ARG0]], [[START2]], [[SIZE2]] : (tensor<6xf64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf64>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]], [[SLICE2]]
    %0, %1, %2 = calc.split %a {split_sizes = array<i64: 1, 2, 3>} : (tensor<6xf64>) -> (tensor<1xf64>, tensor<2xf64>, tensor<3xf64>)
    return %0, %1, %2 : tensor<1xf64>, tensor<2xf64>, tensor<3xf64>
}

// Test 4: 1D i64, single split (trivial)
// CHECK-LABEL: func.func @split_1d_i64_single
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi64>) -> tensor<4xi64>
func.func @split_1d_i64_single(%a: tensor<4xi64>) -> tensor<4xi64> {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<4xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi64>
    // CHECK-NEXT: return [[SLICE0]]
    %0 = calc.split %a {split_sizes = array<i64: 4>} : (tensor<4xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
}

// 2D inputs

// Test 5: 2D f32, split along dim=0
// CHECK-LABEL: func.func @split_2d_f32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x6xf32>) -> (tensor<1x6xf32>, tensor<3x6xf32>)
func.func @split_2d_f32_dim0(%a: tensor<4x6xf32>) -> (tensor<1x6xf32>, tensor<3x6xf32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<4x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x6xf32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[3, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<4x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x6xf32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 1, 3>, dim = 0 : si32} : (tensor<4x6xf32>) -> (tensor<1x6xf32>, tensor<3x6xf32>)
    return %0, %1 : tensor<1x6xf32>, tensor<3x6xf32>
}

// Test 6: 2D i32, split along dim=1
// CHECK-LABEL: func.func @split_2d_i32_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x6xi32>) -> (tensor<3x2xi32>, tensor<3x4xi32>)
func.func @split_2d_i32_dim1(%a: tensor<3x6xi32>) -> (tensor<3x2xi32>, tensor<3x4xi32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<3x6xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x2xi32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<3x6xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x4xi32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = 1 : si32} : (tensor<3x6xi32>) -> (tensor<3x2xi32>, tensor<3x4xi32>)
    return %0, %1 : tensor<3x2xi32>, tensor<3x4xi32>
}

// Test 7: 2D f64, negative dim=-1 (==dim=1)
// CHECK-LABEL: func.func @split_2d_f64_negdim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x6xf64>) -> (tensor<3x2xf64>, tensor<3x4xf64>)
func.func @split_2d_f64_negdim1(%a: tensor<3x6xf64>) -> (tensor<3x2xf64>, tensor<3x4xf64>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<3x6xf64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x2xf64>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<3x6xf64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x4xf64>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = -1 : si32} : (tensor<3x6xf64>) -> (tensor<3x2xf64>, tensor<3x4xf64>)
    return %0, %1 : tensor<3x2xf64>, tensor<3x4xf64>
}

// Test 8: 2D i64, 3-way split along dim=1
// CHECK-LABEL: func.func @split_2d_i64_3way_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x6xi64>) -> (tensor<3x1xi64>, tensor<3x2xi64>, tensor<3x3xi64>)
func.func @split_2d_i64_3way_dim1(%a: tensor<3x6xi64>) -> (tensor<3x1xi64>, tensor<3x2xi64>, tensor<3x3xi64>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<3x6xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x1xi64>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<3x6xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x2xi64>
    // CHECK-DAG: [[START2:%.+]] = tosa.const_shape  {values = dense<[0, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SIZE2:%.+]] = tosa.const_shape  {values = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[SLICE2:%.+]] = tosa.slice [[ARG0]], [[START2]], [[SIZE2]] : (tensor<3x6xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x3xi64>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]], [[SLICE2]]
    %0, %1, %2 = calc.split %a {split_sizes = array<i64: 1, 2, 3>, dim = 1 : si32} : (tensor<3x6xi64>) -> (tensor<3x1xi64>, tensor<3x2xi64>, tensor<3x3xi64>)
    return %0, %1, %2 : tensor<3x1xi64>, tensor<3x2xi64>, tensor<3x3xi64>
}

// 3D inputs

// Test 9: 3D f32, split along dim=0
// CHECK-LABEL: func.func @split_3d_f32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>)
func.func @split_3d_f32_dim0(%a: tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<6x3x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x4xf32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[2, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[4, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<6x3x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x3x4xf32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = 0 : si32} : (tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>)
    return %0, %1 : tensor<2x3x4xf32>, tensor<4x3x4xf32>
}

// Test 10: 3D i32, split along dim=2
// CHECK-LABEL: func.func @split_3d_i32_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x6xi32>) -> (tensor<2x3x2xi32>, tensor<2x3x4xi32>)
func.func @split_3d_i32_dim2(%a: tensor<2x3x6xi32>) -> (tensor<2x3x2xi32>, tensor<2x3x4xi32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 3, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<2x3x6xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x2xi32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 0, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[2, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<2x3x6xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = 2 : si32} : (tensor<2x3x6xi32>) -> (tensor<2x3x2xi32>, tensor<2x3x4xi32>)
    return %0, %1 : tensor<2x3x2xi32>, tensor<2x3x4xi32>
}

// Test 11: 3D f64, negative dim=-2 (==dim=1)
// CHECK-LABEL: func.func @split_3d_f64_negdim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x6x4xf64>) -> (tensor<2x2x4xf64>, tensor<2x4x4xf64>)
func.func @split_3d_f64_negdim2(%a: tensor<2x6x4xf64>) -> (tensor<2x2x4xf64>, tensor<2x4x4xf64>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 2, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<2x6x4xf64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x2x4xf64>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 2, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[2, 4, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<2x6x4xf64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x4x4xf64>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = -2 : si32} : (tensor<2x6x4xf64>) -> (tensor<2x2x4xf64>, tensor<2x4x4xf64>)
    return %0, %1 : tensor<2x2x4xf64>, tensor<2x4x4xf64>
}

// Test 12: 3D i64, 3-way split along dim=1
// CHECK-LABEL: func.func @split_3d_i64_3way_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x6x4xi64>) -> (tensor<2x1x4xi64>, tensor<2x2x4xi64>, tensor<2x3x4xi64>)
func.func @split_3d_i64_3way_dim1(%a: tensor<2x6x4xi64>) -> (tensor<2x1x4xi64>, tensor<2x2x4xi64>, tensor<2x3x4xi64>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<2x6x4xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x1x4xi64>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 1, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[2, 2, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<2x6x4xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x2x4xi64>
    // CHECK-DAG: [[START2:%.+]] = tosa.const_shape  {values = dense<[0, 3, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE2:%.+]] = tosa.const_shape  {values = dense<[2, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE2:%.+]] = tosa.slice [[ARG0]], [[START2]], [[SIZE2]] : (tensor<2x6x4xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x4xi64>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]], [[SLICE2]]
    %0, %1, %2 = calc.split %a {split_sizes = array<i64: 1, 2, 3>, dim = 1 : si32} : (tensor<2x6x4xi64>) -> (tensor<2x1x4xi64>, tensor<2x2x4xi64>, tensor<2x3x4xi64>)
    return %0, %1, %2 : tensor<2x1x4xi64>, tensor<2x2x4xi64>, tensor<2x3x4xi64>
}

// Test 13: 3D f32, negative dim=-3 (==dim=0)
// CHECK-LABEL: func.func @split_3d_f32_negdim3
// CHECK-SAME: ([[ARG0:%.+]]: tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>)
func.func @split_3d_f32_negdim3(%a: tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<6x3x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x4xf32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[2, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[4, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<6x3x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x3x4xf32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>, dim = -3 : si32} : (tensor<6x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x3x4xf32>)
    return %0, %1 : tensor<2x3x4xf32>, tensor<4x3x4xf32>
}

// Test 14: 3D i32, no dim attr (defaults to 0)
// CHECK-LABEL: func.func @split_3d_i32_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<6x3x4xi32>) -> (tensor<2x3x4xi32>, tensor<4x3x4xi32>)
func.func @split_3d_i32_nodim(%a: tensor<6x3x4xi32>) -> (tensor<2x3x4xi32>, tensor<4x3x4xi32>) {
    // CHECK-DAG: [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE0:%.+]] = tosa.const_shape  {values = dense<[2, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE0:%.+]] = tosa.slice [[ARG0]], [[START0]], [[SIZE0]] : (tensor<6x3x4xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x3x4xi32>
    // CHECK-DAG: [[START1:%.+]] = tosa.const_shape  {values = dense<[2, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[4, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[SLICE1:%.+]] = tosa.slice [[ARG0]], [[START1]], [[SIZE1]] : (tensor<6x3x4xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x3x4xi32>
    // CHECK-NEXT: return [[SLICE0]], [[SLICE1]]
    %0, %1 = calc.split %a {split_sizes = array<i64: 2, 4>} : (tensor<6x3x4xi32>) -> (tensor<2x3x4xi32>, tensor<4x3x4xi32>)
    return %0, %1 : tensor<2x3x4xi32>, tensor<4x3x4xi32>
}
