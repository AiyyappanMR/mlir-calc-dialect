// This file verifies the 'OperandsMustbeFloat' trait constraint.

// RUN: not calc-opt %s --calc-to-tosa 2>&1 | FileCheck %s

func.func @test_scalar_pass(%input: f32) -> f32 {
    %result = calc.test %input, %input : f32, f32 -> f32
    return %result : f32
}

func.func @test_scalar_fail(%input: i32) -> i32 {
    // CHECK: error: 'calc.test' op requires all operands to have floating-point element types
    %result = calc.test %input, %input : i32, i32 -> i32
    return %result : i32
}