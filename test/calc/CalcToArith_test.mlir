// RUN: calc-opt --calc-to-arith %s | FileCheck %s

// CHECK-LABEL: func.func @test_add_0
// CHECK-SAME:    ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @test_add_0(%arg0 : i32, %arg1 : i32)-> i32 {
    // CHECK-NEXT: [[RES:%.+]] = arith.addi [[ARG0]], [[ARG1]] : i32
    // CHECK-NEXT: return [[RES]] : i32
    %result = calc.add %arg0, %arg1 : i32
    return %result : i32
}

// CHECK-LABEL: func.func @test_add_1
// CHECK-SAME:    ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64) -> f64
func.func @test_add_1(%arg0 : f64, %arg1 : f64)-> f64 {
    // CHECK-NEXT: [[RES:%.+]] = arith.addf [[ARG0]], [[ARG1]] : f64
    // CHECK-NEXT: return [[RES]] : f64
    %result = calc.add %arg0, %arg1 : f64
    return %result : f64
}
// CHECK-LABEL: func.func @test_mul_0
// CHECK-SAME:    ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @test_mul_0(%arg0 : i32, %arg1 : i32)-> i32 {
    // CHECK-NEXT: [[RES:%.+]] = arith.muli [[ARG0]], [[ARG1]] : i32
    // CHECK-NEXT: return [[RES]] : i32
    %result = calc.mul %arg0, %arg1 : i32
    return %result : i32
}
// CHECK-LABEL: func.func @test_mul_1
// CHECK-SAME:    ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64) -> f64
func.func @test_mul_1(%arg0 : f64, %arg1 : f64)-> f64 {
    // CHECK-NEXT: [[RES:%.+]] = arith.mulf [[ARG0]], [[ARG1]] : f64
    // CHECK-NEXT: return [[RES]] : f64
    %result = calc.mul %arg0, %arg1 : f64
    return %result : f64
}
