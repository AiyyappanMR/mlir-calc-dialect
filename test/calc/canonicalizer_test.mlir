// This file tests canonicalization patterns for the Calc dialect.

// RUN: calc-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_addt_mult_f32
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[A:%.+]]: tensor<4xf32>, [[B:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @fuse_addt_mult_f32(%input: tensor<4xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NEXT: [[RES:%.+]] = calc.addcmul [[INPUT]], [[A]], [[B]] : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: return [[RES]] : tensor<4xf32>
    %mul = calc.mult %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %result = calc.addt %input, %mul : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}
// -----

// CHECK-LABEL: func.func @fuse_addt_mult_f64_2D
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x2xi64>, [[A:%.+]]: tensor<4x2xi64>, [[B:%.+]]: tensor<4x2xi64>) -> tensor<4x2xi64>
func.func @fuse_addt_mult_f64_2D(%input: tensor<4x2xi64>, %a: tensor<4x2xi64>, %b: tensor<4x2xi64>) -> tensor<4x2xi64> {
    // CHECK-NEXT: [[RES:%.+]] = calc.addcmul [[INPUT]], [[A]], [[B]] : (tensor<4x2xi64>, tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    // CHECK-NEXT: return [[RES]] : tensor<4x2xi64>
    %mul = calc.mult %a, %b : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    %result = calc.addt %input, %mul : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi64>
    return %result : tensor<4x2xi64>
}
// -----

// CHECK-LABEL: func.func @no_fuse_mult_multiple_use
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4xf32>, [[A:%.+]]: tensor<4xf32>, [[B:%.+]]: tensor<4xf32>) -> tensor<4xf32>
func.func @no_fuse_mult_multiple_use(%input: tensor<4xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32>{
    // CHECK-NOT: calc.addcmul
    %mul = calc.mult %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %temp = calc.addt %input, %mul : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %res = calc.addt %temp, %mul : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %res : tensor<4xf32>
}
// -----

// CHECK-LABEL: func.func @no_fuse_rhs_not_mult
func.func @no_fuse_rhs_not_mult(%input: tensor<4xf32>, %other: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-NOT: calc.addcmul
    %result = calc.addt %input, %other : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}
