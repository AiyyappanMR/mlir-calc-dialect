// This file contains lit testing of memref-to-linalg pass for memref.copy op.

// RUN: calc-opt %s --memref-to-linalg --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_memref_copy_1d(
// CHECK-SAME:                                %[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: memref<10xf32>) {
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[ARG0]] : memref<10xf32>) outs(%[[ARG1]] : memref<10xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:             linalg.yield %[[IN]] : f32
// CHECK:           }
func.func @test_memref_copy_1d(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  memref.copy %arg0, %arg1 : memref<10xf32> to memref<10xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @test_memref_copy_2d(
// CHECK-SAME:                                %[[ARG0:.*]]: memref<4x8xf64>, %[[ARG1:.*]]: memref<4x8xf64>) {
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[ARG0]] : memref<4x8xf64>) outs(%[[ARG1]] : memref<4x8xf64>) {
// CHECK:           ^bb0(%[[IN:.*]]: f64, %[[OUT:.*]]: f64):
// CHECK:             linalg.yield %[[IN]] : f64
// CHECK:           }
func.func @test_memref_copy_2d(%arg0: memref<4x8xf64>, %arg1: memref<4x8xf64>) {
  memref.copy %arg0, %arg1 : memref<4x8xf64> to memref<4x8xf64>
  return
}

// -----

// CHECK-LABEL:   func.func @test_memref_copy_3d(
// CHECK-SAME:                                %[[ARG0:.*]]: memref<2x3x4xi32>, %[[ARG1:.*]]: memref<2x3x4xi32>) {
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[ARG0]] : memref<2x3x4xi32>) outs(%[[ARG1]] : memref<2x3x4xi32>) {
// CHECK:           ^bb0(%[[IN:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:             linalg.yield %[[IN]] : i32
// CHECK:           }
func.func @test_memref_copy_3d(%arg0: memref<2x3x4xi32>, %arg1: memref<2x3x4xi32>) {
  memref.copy %arg0, %arg1 : memref<2x3x4xi32> to memref<2x3x4xi32>
  return
}

// -----

// CHECK-LABEL:   func.func @test_memref_copy_4d(
// CHECK-SAME:                                %[[ARG0:.*]]: memref<2x3x4x5xi64>, %[[ARG1:.*]]: memref<2x3x4x5xi64>) {
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[ARG0]] : memref<2x3x4x5xi64>) outs(%[[ARG1]] : memref<2x3x4x5xi64>) {
// CHECK:           ^bb0(%[[IN:.*]]: i64, %[[OUT:.*]]: i64):
// CHECK:             linalg.yield %[[IN]] : i64
// CHECK:           }
func.func @test_memref_copy_4d(%arg0: memref<2x3x4x5xi64>, %arg1: memref<2x3x4x5xi64>) {
  memref.copy %arg0, %arg1 : memref<2x3x4x5xi64> to memref<2x3x4x5xi64>
  return
}
