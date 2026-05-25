// This file contains MLIR code that tests the lowering of the `calc.prod` operation to TOSA.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// Test 1: 1D f32 no dim
// CHECK-LABEL: func.func @test_1d_f32_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>) -> tensor<f32>
func.func @test_1d_f32_nodim(%arg0: tensor<4xf32>) -> tensor<f32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xf32>, !tosa.shape<1>) -> tensor<4xf32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[RESHAPE]] {axis = 0 : i32} : (tensor<4xf32>) -> tensor<1xf32>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[EXTRACT:%.+]] = tensor.extract [[REDUCE]][[[C0]]] : tensor<1xf32>
    // CHECK-DAG: [[FROM:%.+]] = tensor.from_elements [[EXTRACT]] : tensor<f32>
    // CHECK-NEXT: return [[FROM]] : tensor<f32>
    %0 = calc.prod %arg0 : (tensor<4xf32>) -> tensor<f32>
    return %0 : tensor<f32>
}

// Test 2: 2D f64 no dim
// CHECK-LABEL: func.func @test_2d_f64_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2xf64>) -> tensor<f64>
func.func @test_2d_f64_nodim(%arg0: tensor<2x2xf64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x2xf64>, !tosa.shape<1>) -> tensor<4xf64>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[RESHAPE]] {axis = 0 : i32} : (tensor<4xf64>) -> tensor<1xf64>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[EXTRACT:%.+]] = tensor.extract [[REDUCE]][[[C0]]] : tensor<1xf64>
    // CHECK-DAG: [[FROM:%.+]] = tensor.from_elements [[EXTRACT]] : tensor<f64>
    // CHECK-NEXT: return [[FROM]] : tensor<f64>
    %0 = calc.prod %arg0 : (tensor<2x2xf64>) -> tensor<f64>
    return %0 : tensor<f64>
}

// Test 3: 1D i32 no dim
// CHECK-LABEL: func.func @test_1d_i32_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi32>) -> tensor<i32>
func.func @test_1d_i32_nodim(%arg0: tensor<4xi32>) -> tensor<i32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xi32>, !tosa.shape<1>) -> tensor<4xi32>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[RESHAPE]] {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1xi32>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[EXTRACT:%.+]] = tensor.extract [[REDUCE]][[[C0]]] : tensor<1xi32>
    // CHECK-DAG: [[FROM:%.+]] = tensor.from_elements [[EXTRACT]] : tensor<i32>
    // CHECK-NEXT: return [[FROM]] : tensor<i32>
    %0 = calc.prod %arg0 : (tensor<4xi32>) -> tensor<i32>
    return %0 : tensor<i32>
}

// Test 4: 2D i64 no dim
// CHECK-LABEL: func.func @test_2d_i64_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xi64>) -> tensor<i64>
func.func @test_2d_i64_nodim(%arg0: tensor<2x3xi64>) -> tensor<i64> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3xi64>, !tosa.shape<1>) -> tensor<6xi64>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[RESHAPE]] {axis = 0 : i32} : (tensor<6xi64>) -> tensor<1xi64>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[EXTRACT:%.+]] = tensor.extract [[REDUCE]][[[C0]]] : tensor<1xi64>
    // CHECK-DAG: [[FROM:%.+]] = tensor.from_elements [[EXTRACT]] : tensor<i64>
    // CHECK-NEXT: return [[FROM]] : tensor<i64>
    %0 = calc.prod %arg0 : (tensor<2x3xi64>) -> tensor<i64>
    return %0 : tensor<i64>
}

// Test 5: 2D f32 dim=0 no keepdim
// CHECK-LABEL: func.func @test_2d_f32_dim0_nokeepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xf32>) -> tensor<3xf32>
func.func @test_2d_f32_dim0_nokeepdim(%arg0: tensor<2x3xf32>) -> tensor<3xf32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 0 : i32} : (tensor<2x3xf32>) -> tensor<1x3xf32>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[REDUCE]], [[SHAPE]] : (tensor<1x3xf32>, !tosa.shape<1>) -> tensor<3xf32>
    // CHECK-NEXT: return [[RESHAPE]] : tensor<3xf32>
    %0 = calc.prod %arg0 {dim = 0 : si32} : (tensor<2x3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
}

// Test 6: 2D i32 dim=1 no keepdim
// CHECK-LABEL: func.func @test_2d_i32_dim1_nokeepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xi32>) -> tensor<2xi32>
func.func @test_2d_i32_dim1_nokeepdim(%arg0: tensor<2x3xi32>) -> tensor<2xi32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[REDUCE]], [[SHAPE]] : (tensor<2x1xi32>, !tosa.shape<1>) -> tensor<2xi32>
    // CHECK-NEXT: return [[RESHAPE]] : tensor<2xi32>
    %0 = calc.prod %arg0 {dim = 1 : si32} : (tensor<2x3xi32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
}

// Test 7: 3D f64 dim=2 no keepdim
// CHECK-LABEL: func.func @test_3d_f64_dim2_nokeepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4x2xf64>) -> tensor<3x4xf64>
func.func @test_3d_f64_dim2_nokeepdim(%arg0: tensor<3x4x2xf64>) -> tensor<3x4xf64> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 2 : i32} : (tensor<3x4x2xf64>) -> tensor<3x4x1xf64>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[REDUCE]], [[SHAPE]] : (tensor<3x4x1xf64>, !tosa.shape<2>) -> tensor<3x4xf64>
    // CHECK-NEXT: return [[RESHAPE]] : tensor<3x4xf64>
    %0 = calc.prod %arg0 {dim = 2 : si32} : (tensor<3x4x2xf64>) -> tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
}

// Test 8: 2D f32 dim=0 keepdim
// CHECK-LABEL: func.func @test_2d_f32_dim0_keepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xf32>) -> tensor<1x3xf32>
func.func @test_2d_f32_dim0_keepdim(%arg0: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 0 : i32} : (tensor<2x3xf32>) -> tensor<1x3xf32>
    // CHECK-NEXT: return [[REDUCE]] : tensor<1x3xf32>
    %0 = calc.prod %arg0 {dim = 0 : si32, keepdim = true} : (tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
}

// Test 9: 2D i32 dim=1 keepdim
// CHECK-LABEL: func.func @test_2d_i32_dim1_keepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xi32>) -> tensor<2x1xi32>
func.func @test_2d_i32_dim1_keepdim(%arg0: tensor<2x3xi32>) -> tensor<2x1xi32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    // CHECK-NEXT: return [[REDUCE]] : tensor<2x1xi32>
    %0 = calc.prod %arg0 {dim = 1 : si32, keepdim = true} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
}

// Test 10: 3D f64 dim=2 keepdim
// CHECK-LABEL: func.func @test_3d_f64_dim2_keepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf64>) -> tensor<2x3x1xf64>
func.func @test_3d_f64_dim2_keepdim(%arg0: tensor<2x3x4xf64>) -> tensor<2x3x1xf64> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 2 : i32} : (tensor<2x3x4xf64>) -> tensor<2x3x1xf64>
    // CHECK-NEXT: return [[REDUCE]] : tensor<2x3x1xf64>
    %0 = calc.prod %arg0 {dim = 2 : si32, keepdim = true} : (tensor<2x3x4xf64>) -> tensor<2x3x1xf64>
    return %0 : tensor<2x3x1xf64>
}

// Test 11: 2D f32 neg dim=-1 no keepdim
// CHECK-LABEL: func.func @test_2d_f32_neg_dim1_nokeepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2xf32>) -> tensor<2xf32>
func.func @test_2d_f32_neg_dim1_nokeepdim(%arg0: tensor<2x2xf32>) -> tensor<2xf32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 1 : i32} : (tensor<2x2xf32>) -> tensor<2x1xf32>
    // CHECK-DAG: [[RESHAPE:%.+]] = tosa.reshape [[REDUCE]], [[SHAPE]] : (tensor<2x1xf32>, !tosa.shape<1>) -> tensor<2xf32>
    // CHECK-NEXT: return [[RESHAPE]] : tensor<2xf32>
    %0 = calc.prod %arg0 {dim = -1 : si32} : (tensor<2x2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}

// Test 12: 2D f32 neg dim=-2 keepdim
// CHECK-LABEL: func.func @test_2d_f32_neg_dim0_keepdim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2xf32>) -> tensor<1x2xf32>
func.func @test_2d_f32_neg_dim0_keepdim(%arg0: tensor<2x2xf32>) -> tensor<1x2xf32> attributes {llvm.emit_c_interface} {
    // CHECK-DAG: [[REDUCE:%.+]] = tosa.reduce_product [[ARG0]] {axis = 0 : i32} : (tensor<2x2xf32>) -> tensor<1x2xf32>
    // CHECK-NEXT: return [[REDUCE]] : tensor<1x2xf32>
    %0 = calc.prod %arg0 {dim = -2 : si32, keepdim = true} : (tensor<2x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
}