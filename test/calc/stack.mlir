// This file contains MLIR code to test the lowering of calc.stackOp to tosa.concatOp.

// RUN: calc-opt %s --calc-to-tosa | FileCheck %s

// 1D inputs

// Test 1: 1D i32, 1 input, dim=0 
// CHECK-LABEL: func.func @stack_1d_1inp_i32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi32>) -> tensor<1x4xi32>
func.func @stack_1d_1inp_i32_dim0(%a: tensor<4xi32>) -> tensor<1x4xi32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]] {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x4xi32>
    // CHECK-NEXT: return
    %0 = calc.stack %a {dim = 0 : si32} : (tensor<4xi32>) -> tensor<1x4xi32>
    return %0 : tensor<1x4xi32>
}

// Test 2: 1D f32, 2 inputs, dim=0  
// CHECK-LABEL: func.func @stack_1d_2inp_f32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<2x4xf32>
func.func @stack_1d_2inp_f32_dim0(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<2x4xf32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 0 : i32} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b {dim = 0 : si32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
}

// Test 3: 1D i64, 3 inputs, dim=1  
// CHECK-LABEL: func.func @stack_1d_3inp_i64_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi64>, [[ARG1:%.+]]: tensor<4xi64>, [[ARG2:%.+]]: tensor<4xi64>) -> tensor<4x3xi64>
func.func @stack_1d_3inp_i64_dim1(%a: tensor<4xi64>, %b: tensor<4xi64>, %c: tensor<4xi64>) -> tensor<4x3xi64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xi64>, !tosa.shape<2>) -> tensor<4x1xi64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<4xi64>, !tosa.shape<2>) -> tensor<4x1xi64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<4xi64>, !tosa.shape<2>) -> tensor<4x1xi64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]] {axis = 1 : i32} : (tensor<4x1xi64>, tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x3xi64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c {dim = 1 : si32} : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x3xi64>
    return %0 : tensor<4x3xi64>
}

// Test 4: 1D f64, 4 inputs, neg dim=-1 
// CHECK-LABEL: func.func @stack_1d_4inp_f64_neg_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf64>, [[ARG1:%.+]]: tensor<4xf64>, [[ARG2:%.+]]: tensor<4xf64>, [[ARG3:%.+]]: tensor<4xf64>) -> tensor<4x4xf64>
func.func @stack_1d_4inp_f64_neg_dim1(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>, %d: tensor<4xf64>) -> tensor<4x4xf64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<4xf64>, !tosa.shape<2>) -> tensor<4x1xf64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<4xf64>, !tosa.shape<2>) -> tensor<4x1xf64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<4xf64>, !tosa.shape<2>) -> tensor<4x1xf64>
    // CHECK-DAG: [[R3:%.+]] = tosa.reshape [[ARG3]], [[SHAPE]] : (tensor<4xf64>, !tosa.shape<2>) -> tensor<4x1xf64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]], [[R3]] {axis = 1 : i32} : (tensor<4x1xf64>, tensor<4x1xf64>, tensor<4x1xf64>, tensor<4x1xf64>) -> tensor<4x4xf64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c, %d {dim = -1 : si32} : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
}

// 2D inputs

// Test 5: 2D f32, 1 input, dim=0 
// CHECK-LABEL: func.func @stack_2d_1inp_f32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4xf32>) -> tensor<1x3x4xf32>
func.func @stack_2d_1inp_f32_dim0(%a: tensor<3x4xf32>) -> tensor<1x3x4xf32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]] {axis = 0 : i32} : (tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    // CHECK-NEXT: return
    %0 = calc.stack %a {dim = 0 : si32} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    return %0 : tensor<1x3x4xf32>
}

// Test 6: 2D i32, 2 inputs, dim=1 
// CHECK-LABEL: func.func @stack_2d_2inp_i32_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4xi32>, [[ARG1:%.+]]: tensor<3x4xi32>) -> tensor<3x2x4xi32>
func.func @stack_2d_2inp_i32_dim1(%a: tensor<3x4xi32>, %b: tensor<3x4xi32>) -> tensor<3x2x4xi32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[3, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<3x1x4xi32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<3x1x4xi32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 1 : i32} : (tensor<3x1x4xi32>, tensor<3x1x4xi32>) -> tensor<3x2x4xi32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b {dim = 1 : si32} : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x2x4xi32>
    return %0 : tensor<3x2x4xi32>
}

// Test 7: 2D f64, 3 inputs, dim=2 
// CHECK-LABEL: func.func @stack_2d_3inp_f64_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4xf64>, [[ARG1:%.+]]: tensor<3x4xf64>, [[ARG2:%.+]]: tensor<3x4xf64>) -> tensor<3x4x3xf64>
func.func @stack_2d_3inp_f64_dim2(%a: tensor<3x4xf64>, %b: tensor<3x4xf64>, %c: tensor<3x4xf64>) -> tensor<3x4x3xf64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[3, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<3x4xf64>, !tosa.shape<3>) -> tensor<3x4x1xf64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<3x4xf64>, !tosa.shape<3>) -> tensor<3x4x1xf64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<3x4xf64>, !tosa.shape<3>) -> tensor<3x4x1xf64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]] {axis = 2 : i32} : (tensor<3x4x1xf64>, tensor<3x4x1xf64>, tensor<3x4x1xf64>) -> tensor<3x4x3xf64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c {dim = 2 : si32} : (tensor<3x4xf64>, tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x4x3xf64>
    return %0 : tensor<3x4x3xf64>
}

// Test 8: 2D i64, 4 inputs, neg dim=-1 
// CHECK-LABEL: func.func @stack_2d_4inp_i64_neg_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4xi64>, [[ARG1:%.+]]: tensor<3x4xi64>, [[ARG2:%.+]]: tensor<3x4xi64>, [[ARG3:%.+]]: tensor<3x4xi64>) -> tensor<3x4x4xi64>
func.func @stack_2d_4inp_i64_neg_dim1(%a: tensor<3x4xi64>, %b: tensor<3x4xi64>, %c: tensor<3x4xi64>, %d: tensor<3x4xi64>) -> tensor<3x4x4xi64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[3, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<3x4xi64>, !tosa.shape<3>) -> tensor<3x4x1xi64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<3x4xi64>, !tosa.shape<3>) -> tensor<3x4x1xi64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<3x4xi64>, !tosa.shape<3>) -> tensor<3x4x1xi64>
    // CHECK-DAG: [[R3:%.+]] = tosa.reshape [[ARG3]], [[SHAPE]] : (tensor<3x4xi64>, !tosa.shape<3>) -> tensor<3x4x1xi64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]], [[R3]] {axis = 2 : i32} : (tensor<3x4x1xi64>, tensor<3x4x1xi64>, tensor<3x4x1xi64>, tensor<3x4x1xi64>) -> tensor<3x4x4xi64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c, %d {dim = -1 : si32} : (tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>) -> tensor<3x4x4xi64>
    return %0 : tensor<3x4x4xi64>
}

// Test 9: 2D i32, 2 inputs, no dim attr 
// CHECK-LABEL: func.func @stack_2d_2inp_i32_nodim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4xi32>, [[ARG1:%.+]]: tensor<3x4xi32>) -> tensor<2x3x4xi32>
func.func @stack_2d_2inp_i32_nodim(%a: tensor<3x4xi32>, %b: tensor<3x4xi32>) -> tensor<2x3x4xi32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<1x3x4xi32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<3x4xi32>, !tosa.shape<3>) -> tensor<1x3x4xi32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 0 : i32} : (tensor<1x3x4xi32>, tensor<1x3x4xi32>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
}

// 3D inputs

// Test 10: 3D i32, 1 input, dim=0 
// CHECK-LABEL: func.func @stack_3d_1inp_i32_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xi32>) -> tensor<1x2x3x4xi32>
func.func @stack_3d_1inp_i32_dim0(%a: tensor<2x3x4xi32>) -> tensor<1x2x3x4xi32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4xi32>, !tosa.shape<4>) -> tensor<1x2x3x4xi32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]] {axis = 0 : i32} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
    // CHECK-NEXT: return
    %0 = calc.stack %a {dim = 0 : si32} : (tensor<2x3x4xi32>) -> tensor<1x2x3x4xi32>
    return %0 : tensor<1x2x3x4xi32>
}

// Test 11: 3D f32, 2 inputs, dim=1 
// CHECK-LABEL: func.func @stack_3d_2inp_f32_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf32>, [[ARG1:%.+]]: tensor<2x3x4xf32>) -> tensor<2x2x3x4xf32>
func.func @stack_3d_2inp_f32_dim1(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>) -> tensor<2x2x3x4xf32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 1, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4xf32>, !tosa.shape<4>) -> tensor<2x1x3x4xf32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4xf32>, !tosa.shape<4>) -> tensor<2x1x3x4xf32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 1 : i32} : (tensor<2x1x3x4xf32>, tensor<2x1x3x4xf32>) -> tensor<2x2x3x4xf32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b {dim = 1 : si32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x2x3x4xf32>
    return %0 : tensor<2x2x3x4xf32>
}

// Test 12: 3D i64, 3 inputs, dim=2  
// CHECK-LABEL: func.func @stack_3d_3inp_i64_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xi64>, [[ARG1:%.+]]: tensor<2x3x4xi64>, [[ARG2:%.+]]: tensor<2x3x4xi64>) -> tensor<2x3x3x4xi64>
func.func @stack_3d_3inp_i64_dim2(%a: tensor<2x3x4xi64>, %b: tensor<2x3x4xi64>, %c: tensor<2x3x4xi64>) -> tensor<2x3x3x4xi64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 1, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4xi64>, !tosa.shape<4>) -> tensor<2x3x1x4xi64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4xi64>, !tosa.shape<4>) -> tensor<2x3x1x4xi64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<2x3x4xi64>, !tosa.shape<4>) -> tensor<2x3x1x4xi64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]] {axis = 2 : i32} : (tensor<2x3x1x4xi64>, tensor<2x3x1x4xi64>, tensor<2x3x1x4xi64>) -> tensor<2x3x3x4xi64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c {dim = 2 : si32} : (tensor<2x3x4xi64>, tensor<2x3x4xi64>, tensor<2x3x4xi64>) -> tensor<2x3x3x4xi64>
    return %0 : tensor<2x3x3x4xi64>
}

// Test 13: 3D f64, 4 inputs, dim=3 
// CHECK-LABEL: func.func @stack_3d_4inp_f64_dim3
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf64>, [[ARG1:%.+]]: tensor<2x3x4xf64>, [[ARG2:%.+]]: tensor<2x3x4xf64>, [[ARG3:%.+]]: tensor<2x3x4xf64>) -> tensor<2x3x4x4xf64>
func.func @stack_3d_4inp_f64_dim3(%a: tensor<2x3x4xf64>, %b: tensor<2x3x4xf64>, %c: tensor<2x3x4xf64>, %d: tensor<2x3x4xf64>) -> tensor<2x3x4x4xf64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 4, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4xf64>, !tosa.shape<4>) -> tensor<2x3x4x1xf64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4xf64>, !tosa.shape<4>) -> tensor<2x3x4x1xf64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<2x3x4xf64>, !tosa.shape<4>) -> tensor<2x3x4x1xf64>
    // CHECK-DAG: [[R3:%.+]] = tosa.reshape [[ARG3]], [[SHAPE]] : (tensor<2x3x4xf64>, !tosa.shape<4>) -> tensor<2x3x4x1xf64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]], [[R3]] {axis = 3 : i32} : (tensor<2x3x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x3x4x1xf64>) -> tensor<2x3x4x4xf64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c, %d {dim = 3 : si32} : (tensor<2x3x4xf64>, tensor<2x3x4xf64>, tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4x4xf64>
    return %0 : tensor<2x3x4x4xf64>
}

// Test 14: 3D f32, 2 inputs, neg dim=-2 (==dim=2)  
// CHECK-LABEL: func.func @stack_3d_2inp_f32_neg_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf32>, [[ARG1:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x2x4xf32>
func.func @stack_3d_2inp_f32_neg_dim2(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>) -> tensor<2x3x2x4xf32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 1, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4xf32>, !tosa.shape<4>) -> tensor<2x3x1x4xf32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4xf32>, !tosa.shape<4>) -> tensor<2x3x1x4xf32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 2 : i32} : (tensor<2x3x1x4xf32>, tensor<2x3x1x4xf32>) -> tensor<2x3x2x4xf32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b {dim = -2 : si32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x2x4xf32>
    return %0 : tensor<2x3x2x4xf32>
}

// 4D inputs 

// Test 15: 4D f64, 1 input, dim=2  
// CHECK-LABEL: func.func @stack_4d_1inp_f64_dim2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf64>) -> tensor<2x3x1x4x5xf64>
func.func @stack_4d_1inp_f64_dim2(%a: tensor<2x3x4x5xf64>) -> tensor<2x3x1x4x5xf64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 1, 4, 5]> : tensor<5xindex>} : () -> !tosa.shape<5>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4x5xf64>, !tosa.shape<5>) -> tensor<2x3x1x4x5xf64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]] {axis = 2 : i32} : (tensor<2x3x1x4x5xf64>) -> tensor<2x3x1x4x5xf64>
    // CHECK-NEXT: return
    %0 = calc.stack %a {dim = 2 : si32} : (tensor<2x3x4x5xf64>) -> tensor<2x3x1x4x5xf64>
    return %0 : tensor<2x3x1x4x5xf64>
}

// Test 16: 4D i32, 2 inputs, dim=1  
// CHECK-LABEL: func.func @stack_4d_2inp_i32_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xi32>, [[ARG1:%.+]]: tensor<2x3x4x5xi32>) -> tensor<2x2x3x4x5xi32>
func.func @stack_4d_2inp_i32_dim1(%a: tensor<2x3x4x5xi32>, %b: tensor<2x3x4x5xi32>) -> tensor<2x2x3x4x5xi32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 1, 3, 4, 5]> : tensor<5xindex>} : () -> !tosa.shape<5>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4x5xi32>, !tosa.shape<5>) -> tensor<2x1x3x4x5xi32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4x5xi32>, !tosa.shape<5>) -> tensor<2x1x3x4x5xi32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]] {axis = 1 : i32} : (tensor<2x1x3x4x5xi32>, tensor<2x1x3x4x5xi32>) -> tensor<2x2x3x4x5xi32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b {dim = 1 : si32} : (tensor<2x3x4x5xi32>, tensor<2x3x4x5xi32>) -> tensor<2x2x3x4x5xi32>
    return %0 : tensor<2x2x3x4x5xi32>
}

// Test 17: 4D f32, 3 inputs, dim=4 
// CHECK-LABEL: func.func @stack_4d_3inp_f32_dim4
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf32>, [[ARG1:%.+]]: tensor<2x3x4x5xf32>, [[ARG2:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5x3xf32>
func.func @stack_4d_3inp_f32_dim4(%a: tensor<2x3x4x5xf32>, %b: tensor<2x3x4x5xf32>, %c: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5x3xf32> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[2, 3, 4, 5, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4x5xf32>, !tosa.shape<5>) -> tensor<2x3x4x5x1xf32>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4x5xf32>, !tosa.shape<5>) -> tensor<2x3x4x5x1xf32>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<2x3x4x5xf32>, !tosa.shape<5>) -> tensor<2x3x4x5x1xf32>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]] {axis = 4 : i32} : (tensor<2x3x4x5x1xf32>, tensor<2x3x4x5x1xf32>, tensor<2x3x4x5x1xf32>) -> tensor<2x3x4x5x3xf32>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c {dim = 4 : si32} : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5x3xf32>
    return %0 : tensor<2x3x4x5x3xf32>
}

// Test 18: 4D i64, 4 inputs, dim=0  
// CHECK-LABEL: func.func @stack_4d_4inp_i64_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xi64>, [[ARG1:%.+]]: tensor<2x3x4x5xi64>, [[ARG2:%.+]]: tensor<2x3x4x5xi64>, [[ARG3:%.+]]: tensor<2x3x4x5xi64>) -> tensor<4x2x3x4x5xi64>
func.func @stack_4d_4inp_i64_dim0(%a: tensor<2x3x4x5xi64>, %b: tensor<2x3x4x5xi64>, %c: tensor<2x3x4x5xi64>, %d: tensor<2x3x4x5xi64>) -> tensor<4x2x3x4x5xi64> {
    // CHECK-DAG: [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 2, 3, 4, 5]> : tensor<5xindex>} : () -> !tosa.shape<5>
    // CHECK-DAG: [[R0:%.+]] = tosa.reshape [[ARG0]], [[SHAPE]] : (tensor<2x3x4x5xi64>, !tosa.shape<5>) -> tensor<1x2x3x4x5xi64>
    // CHECK-DAG: [[R1:%.+]] = tosa.reshape [[ARG1]], [[SHAPE]] : (tensor<2x3x4x5xi64>, !tosa.shape<5>) -> tensor<1x2x3x4x5xi64>
    // CHECK-DAG: [[R2:%.+]] = tosa.reshape [[ARG2]], [[SHAPE]] : (tensor<2x3x4x5xi64>, !tosa.shape<5>) -> tensor<1x2x3x4x5xi64>
    // CHECK-DAG: [[R3:%.+]] = tosa.reshape [[ARG3]], [[SHAPE]] : (tensor<2x3x4x5xi64>, !tosa.shape<5>) -> tensor<1x2x3x4x5xi64>
    // CHECK-DAG: {{%.+}} = tosa.concat [[R0]], [[R1]], [[R2]], [[R3]] {axis = 0 : i32} : (tensor<1x2x3x4x5xi64>, tensor<1x2x3x4x5xi64>, tensor<1x2x3x4x5xi64>, tensor<1x2x3x4x5xi64>) -> tensor<4x2x3x4x5xi64>
    // CHECK-NEXT: return
    %0 = calc.stack %a, %b, %c, %d {dim = 0 : si32} : (tensor<2x3x4x5xi64>, tensor<2x3x4x5xi64>, tensor<2x3x4x5xi64>, tensor<2x3x4x5xi64>) -> tensor<4x2x3x4x5xi64>
    return %0 : tensor<4x2x3x4x5xi64>
}
