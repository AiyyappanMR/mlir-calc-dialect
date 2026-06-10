// RUN: calc-opt %s --calc-to-tosa | FileCheck %s

// 1D inputs

// Test 1: 1D f32, 1+1 inputs, dim=0, no scale
// CHECK-LABEL: func.func @catmuladd_1d_1p1inp_f32_dim0_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @catmuladd_1d_1p1inp_f32_dim0_noscale(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]] {axis = 0 : i32} : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG1]] {axis = 0 : i32} : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b) {operandSegmentSizes = array<i32: 1, 1, 0>, dim = 0 : si32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// Test 2: 1D f32, 2+2 inputs, dim=0, no scale
// CHECK-LABEL: func.func @catmuladd_1d_2p2inp_f32_dim0_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<2xf32>, [[ARG2:%.+]]: tensor<2xf32>, [[ARG3:%.+]]: tensor<2xf32>) -> tensor<4xf32>
func.func @catmuladd_1d_2p2inp_f32_dim0_noscale(%a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>, %d: tensor<2xf32>) -> tensor<4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = 0 : si32} : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// Test 3: 1D i32, 2+2 inputs, dim=0, with scale
// CHECK-LABEL: func.func @catmuladd_1d_2p2inp_i32_dim0_scale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARG2:%.+]]: tensor<2xi32>, [[ARG3:%.+]]: tensor<2xi32>, [[ARG4:%.+]]: tensor<4xi32>) -> tensor<4xi32>
func.func @catmuladd_1d_2p2inp_i32_dim0_scale(%a: tensor<2xi32>, %b: tensor<2xi32>, %c: tensor<2xi32>, %d: tensor<2xi32>, %s: tensor<4xi32>) -> tensor<4xi32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 0 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 0 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[MUL]], [[ARG4]] : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    // CHECK-NEXT: return [[ADD]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %s) {operandSegmentSizes = array<i32: 2, 2, 1>, dim = 0 : si32} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
}

// Test 4: 1D f64, 3+3 inputs, no dim attr, no scale
// CHECK-LABEL: func.func @catmuladd_1d_3p3inp_f64_nodim_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2xf64>, [[ARG1:%.+]]: tensor<2xf64>, [[ARG2:%.+]]: tensor<2xf64>, [[ARG3:%.+]]: tensor<2xf64>, [[ARG4:%.+]]: tensor<2xf64>, [[ARG5:%.+]]: tensor<2xf64>) -> tensor<6xf64>
func.func @catmuladd_1d_3p3inp_f64_nodim_noscale(%a: tensor<2xf64>, %b: tensor<2xf64>, %c: tensor<2xf64>, %d: tensor<2xf64>, %e: tensor<2xf64>, %f: tensor<2xf64>) -> tensor<6xf64> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]], [[ARG2]] {axis = 0 : i32} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<6xf64>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG3]], [[ARG4]], [[ARG5]] {axis = 0 : i32} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<6xf64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<6xf64>, tensor<6xf64>, tensor<1xi8>) -> tensor<6xf64>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %e, %f) {operandSegmentSizes = array<i32: 3, 3, 0>} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<6xf64>
    return %0 : tensor<6xf64>
}

// 2D inputs

// Test 5: 2D f32, 2+2 inputs, dim=0, no scale
// CHECK-LABEL: func.func @catmuladd_2d_2p2inp_f32_dim0_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4xf32>, [[ARG1:%.+]]: tensor<2x4xf32>, [[ARG2:%.+]]: tensor<2x4xf32>, [[ARG3:%.+]]: tensor<2x4xf32>) -> tensor<4x4xf32>
func.func @catmuladd_2d_2p2inp_f32_dim0_noscale(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>, %c: tensor<2x4xf32>, %d: tensor<2x4xf32>) -> tensor<4x4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 0 : i32} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 0 : i32} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<1xi8>) -> tensor<4x4xf32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = 0 : si32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
}

// Test 6: 2D i32, 2+2 inputs, dim=1, no scale
// CHECK-LABEL: func.func @catmuladd_2d_2p2inp_i32_dim1_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x2xi32>, [[ARG1:%.+]]: tensor<3x2xi32>, [[ARG2:%.+]]: tensor<3x2xi32>, [[ARG3:%.+]]: tensor<3x2xi32>) -> tensor<3x4xi32>
func.func @catmuladd_2d_2p2inp_i32_dim1_noscale(%a: tensor<3x2xi32>, %b: tensor<3x2xi32>, %c: tensor<3x2xi32>, %d: tensor<3x2xi32>) -> tensor<3x4xi32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 1 : i32} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x4xi32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 1 : i32} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x4xi32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<3x4xi32>, tensor<3x4xi32>, tensor<1xi8>) -> tensor<3x4xi32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = 1 : si32} : (tensor<3x2xi32>, tensor<3x2xi32>, tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x4xi32>
    return %0 : tensor<3x4xi32>
}

// Test 7: 2D f64, 2+2 inputs, neg dim=-1 (==dim=1), no scale
// CHECK-LABEL: func.func @catmuladd_2d_2p2inp_f64_negdim1_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x2xf64>, [[ARG1:%.+]]: tensor<3x2xf64>, [[ARG2:%.+]]: tensor<3x2xf64>, [[ARG3:%.+]]: tensor<3x2xf64>) -> tensor<3x4xf64>
func.func @catmuladd_2d_2p2inp_f64_negdim1_noscale(%a: tensor<3x2xf64>, %b: tensor<3x2xf64>, %c: tensor<3x2xf64>, %d: tensor<3x2xf64>) -> tensor<3x4xf64> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 1 : i32} : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x4xf64>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 1 : i32} : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x4xf64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<3x4xf64>, tensor<3x4xf64>, tensor<1xi8>) -> tensor<3x4xf64>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = -1 : si32} : (tensor<3x2xf64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
}

// Test 8: 2D f32, 2+2 inputs, dim=1, with scale
// CHECK-LABEL: func.func @catmuladd_2d_2p2inp_f32_dim1_scale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x2xf32>, [[ARG1:%.+]]: tensor<3x2xf32>, [[ARG2:%.+]]: tensor<3x2xf32>, [[ARG3:%.+]]: tensor<3x2xf32>, [[ARG4:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32>
func.func @catmuladd_2d_2p2inp_f32_dim1_scale(%a: tensor<3x2xf32>, %b: tensor<3x2xf32>, %c: tensor<3x2xf32>, %d: tensor<3x2xf32>, %s: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 1 : i32} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 1 : i32} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[MUL]], [[ARG4]] : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    // CHECK-NEXT: return [[ADD]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %s) {operandSegmentSizes = array<i32: 2, 2, 1>, dim = 1 : si32} : (tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
}

// Test 9: 2D i64, 3+3 inputs, dim=0, with broadcast scale
// CHECK-LABEL: func.func @catmuladd_2d_3p3inp_i64_dim0_bcastscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4xi64>, [[ARG1:%.+]]: tensor<2x4xi64>, [[ARG2:%.+]]: tensor<2x4xi64>, [[ARG3:%.+]]: tensor<2x4xi64>, [[ARG4:%.+]]: tensor<2x4xi64>, [[ARG5:%.+]]: tensor<2x4xi64>, [[ARG6:%.+]]: tensor<1x4xi64>) -> tensor<6x4xi64>
func.func @catmuladd_2d_3p3inp_i64_dim0_bcastscale(%a: tensor<2x4xi64>, %b: tensor<2x4xi64>, %c: tensor<2x4xi64>, %d: tensor<2x4xi64>, %e: tensor<2x4xi64>, %f: tensor<2x4xi64>, %s: tensor<1x4xi64>) -> tensor<6x4xi64> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]], [[ARG2]] {axis = 0 : i32} : (tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>) -> tensor<6x4xi64>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG3]], [[ARG4]], [[ARG5]] {axis = 0 : i32} : (tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>) -> tensor<6x4xi64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<6x4xi64>, tensor<6x4xi64>, tensor<1xi8>) -> tensor<6x4xi64>
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[MUL]], [[ARG6]] : (tensor<6x4xi64>, tensor<1x4xi64>) -> tensor<6x4xi64>
    // CHECK-NEXT: return [[ADD]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %e, %f, %s) {operandSegmentSizes = array<i32: 3, 3, 1>, dim = 0 : si32} : (tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>, tensor<1x4xi64>) -> tensor<6x4xi64>
    return %0 : tensor<6x4xi64>
}

// 3D inputs

// Test 10: 3D f32, 2+2 inputs, dim=0, no scale
// CHECK-LABEL: func.func @catmuladd_3d_2p2inp_f32_dim0_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4xf32>, [[ARG1:%.+]]: tensor<2x3x4xf32>, [[ARG2:%.+]]: tensor<2x3x4xf32>, [[ARG3:%.+]]: tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
func.func @catmuladd_3d_2p2inp_f32_dim0_noscale(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>, %c: tensor<2x3x4xf32>, %d: tensor<2x3x4xf32>) -> tensor<4x3x4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 0 : i32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 0 : i32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<4x3x4xf32>, tensor<4x3x4xf32>, tensor<1xi8>) -> tensor<4x3x4xf32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = 0 : si32} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
    return %0 : tensor<4x3x4xf32>
}

// Test 11: 3D i32, 2+2 inputs, dim=2, no scale
// CHECK-LABEL: func.func @catmuladd_3d_2p2inp_i32_dim2_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x2xi32>, [[ARG1:%.+]]: tensor<2x3x2xi32>, [[ARG2:%.+]]: tensor<2x3x2xi32>, [[ARG3:%.+]]: tensor<2x3x2xi32>) -> tensor<2x3x4xi32>
func.func @catmuladd_3d_2p2inp_i32_dim2_noscale(%a: tensor<2x3x2xi32>, %b: tensor<2x3x2xi32>, %c: tensor<2x3x2xi32>, %d: tensor<2x3x2xi32>) -> tensor<2x3x4xi32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 2 : i32} : (tensor<2x3x2xi32>, tensor<2x3x2xi32>) -> tensor<2x3x4xi32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 2 : i32} : (tensor<2x3x2xi32>, tensor<2x3x2xi32>) -> tensor<2x3x4xi32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<2x3x4xi32>, tensor<2x3x4xi32>, tensor<1xi8>) -> tensor<2x3x4xi32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = 2 : si32} : (tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
}

// Test 12: 3D f64, 2+2 inputs, neg dim=-2 (==dim=1), no scale
// CHECK-LABEL: func.func @catmuladd_3d_2p2inp_f64_negdim2_noscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2x4xf64>, [[ARG1:%.+]]: tensor<2x2x4xf64>, [[ARG2:%.+]]: tensor<2x2x4xf64>, [[ARG3:%.+]]: tensor<2x2x4xf64>) -> tensor<2x4x4xf64>
func.func @catmuladd_3d_2p2inp_f64_negdim2_noscale(%a: tensor<2x2x4xf64>, %b: tensor<2x2x4xf64>, %c: tensor<2x2x4xf64>, %d: tensor<2x2x4xf64>) -> tensor<2x4x4xf64> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]] {axis = 1 : i32} : (tensor<2x2x4xf64>, tensor<2x2x4xf64>) -> tensor<2x4x4xf64>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG2]], [[ARG3]] {axis = 1 : i32} : (tensor<2x2x4xf64>, tensor<2x2x4xf64>) -> tensor<2x4x4xf64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<1xi8>) -> tensor<2x4x4xf64>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d) {operandSegmentSizes = array<i32: 2, 2, 0>, dim = -2 : si32} : (tensor<2x2x4xf64>, tensor<2x2x4xf64>, tensor<2x2x4xf64>, tensor<2x2x4xf64>) -> tensor<2x4x4xf64>
    return %0 : tensor<2x4x4xf64>
}

// Test 13: 3D f32, 3+3 inputs, dim=1, with scale
// CHECK-LABEL: func.func @catmuladd_3d_3p3inp_f32_dim1_scale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2x4xf32>, [[ARG1:%.+]]: tensor<2x2x4xf32>, [[ARG2:%.+]]: tensor<2x2x4xf32>, [[ARG3:%.+]]: tensor<2x2x4xf32>, [[ARG4:%.+]]: tensor<2x2x4xf32>, [[ARG5:%.+]]: tensor<2x2x4xf32>, [[ARG6:%.+]]: tensor<2x6x4xf32>) -> tensor<2x6x4xf32>
func.func @catmuladd_3d_3p3inp_f32_dim1_scale(%a: tensor<2x2x4xf32>, %b: tensor<2x2x4xf32>, %c: tensor<2x2x4xf32>, %d: tensor<2x2x4xf32>, %e: tensor<2x2x4xf32>, %f: tensor<2x2x4xf32>, %s: tensor<2x6x4xf32>) -> tensor<2x6x4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]], [[ARG2]] {axis = 1 : i32} : (tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x6x4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG3]], [[ARG4]], [[ARG5]] {axis = 1 : i32} : (tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x6x4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<2x6x4xf32>, tensor<2x6x4xf32>, tensor<1xi8>) -> tensor<2x6x4xf32>
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[MUL]], [[ARG6]] : (tensor<2x6x4xf32>, tensor<2x6x4xf32>) -> tensor<2x6x4xf32>
    // CHECK-NEXT: return [[ADD]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %e, %f, %s) {operandSegmentSizes = array<i32: 3, 3, 1>, dim = 1 : si32} : (tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x6x4xf32>) -> tensor<2x6x4xf32>
    return %0 : tensor<2x6x4xf32>
}

// Test 14: 3D i64, 4+4 inputs, dim=2, with broadcast scale
// CHECK-LABEL: func.func @catmuladd_3d_4p4inp_i64_dim2_bcastscale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x2xi64>, [[ARG1:%.+]]: tensor<2x3x2xi64>, [[ARG2:%.+]]: tensor<2x3x2xi64>, [[ARG3:%.+]]: tensor<2x3x2xi64>, [[ARG4:%.+]]: tensor<2x3x2xi64>, [[ARG5:%.+]]: tensor<2x3x2xi64>, [[ARG6:%.+]]: tensor<2x3x2xi64>, [[ARG7:%.+]]: tensor<2x3x2xi64>, [[ARG8:%.+]]: tensor<1x1x8xi64>) -> tensor<2x3x8xi64>
func.func @catmuladd_3d_4p4inp_i64_dim2_bcastscale(%a: tensor<2x3x2xi64>, %b: tensor<2x3x2xi64>, %c: tensor<2x3x2xi64>, %d: tensor<2x3x2xi64>, %e: tensor<2x3x2xi64>, %f: tensor<2x3x2xi64>, %g: tensor<2x3x2xi64>, %h: tensor<2x3x2xi64>, %s: tensor<1x1x8xi64>) -> tensor<2x3x8xi64> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]] {axis = 2 : i32} : (tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x8xi64>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]] {axis = 2 : i32} : (tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x8xi64>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<2x3x8xi64>, tensor<2x3x8xi64>, tensor<1xi8>) -> tensor<2x3x8xi64>
    // CHECK-DAG: [[ADD:%.+]] = tosa.add [[MUL]], [[ARG8]] : (tensor<2x3x8xi64>, tensor<1x1x8xi64>) -> tensor<2x3x8xi64>
    // CHECK-NEXT: return [[ADD]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %e, %f, %g, %h, %s) {operandSegmentSizes = array<i32: 4, 4, 1>, dim = 2 : si32} : (tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2xi64>, tensor<1x1x8xi64>) -> tensor<2x3x8xi64>
    return %0 : tensor<2x3x8xi64>
}

// Test 15: 2D f32, 3+2 inputs, dim=0, no scale (unequal groups)
// CHECK-LABEL: func.func @catmuladd_2d_3p2inp_f32_dim0_noscale_unequal
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4xf32>, [[ARG1:%.+]]: tensor<2x4xf32>, [[ARG2:%.+]]: tensor<2x4xf32>, [[ARG3:%.+]]: tensor<3x4xf32>, [[ARG4:%.+]]: tensor<3x4xf32>) -> tensor<6x4xf32>
func.func @catmuladd_2d_3p2inp_f32_dim0_noscale_unequal(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>, %c: tensor<2x4xf32>, %d: tensor<3x4xf32>, %e: tensor<3x4xf32>) -> tensor<6x4xf32> {
    // CHECK-DAG: [[CAT1:%.+]] = tosa.concat [[ARG0]], [[ARG1]], [[ARG2]] {axis = 0 : i32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<6x4xf32>
    // CHECK-DAG: [[CAT2:%.+]] = tosa.concat [[ARG3]], [[ARG4]] {axis = 0 : i32} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<6x4xf32>
    // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"()
    // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[CAT1]], [[CAT2]], [[SHIFT]] : (tensor<6x4xf32>, tensor<6x4xf32>, tensor<1xi8>) -> tensor<6x4xf32>
    // CHECK-NEXT: return [[MUL]]
    %0 = calc.catmuladd (%a, %b, %c, %d, %e) {operandSegmentSizes = array<i32: 3, 2, 0>, dim = 0 : si32} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<6x4xf32>
    return %0 : tensor<6x4xf32>
}