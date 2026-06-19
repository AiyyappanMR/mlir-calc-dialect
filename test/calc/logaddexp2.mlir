// Tests lowering of calc.logaddexp2 to tosa dialect.

// RUN: calc-opt --calc-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @test_logaddexp2_0d
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f64>, [[ARG1:%.+]]: tensor<f64>) -> tensor<f64>
func.func @test_logaddexp2_0d(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
  // CHECK-DAG: [[POW1:%.+]] = tosa.pow [[TWO]], [[ARG0]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[POW2:%.+]] = tosa.pow [[TWO]], [[ARG1]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[ADD:%.+]] = tosa.add [[POW1]], [[POW2]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[LOG_ADD:%.+]] = tosa.log [[ADD]] : (tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[LOG2:%.+]] = tosa.log [[TWO]] : (tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[RECIP_LOG2:%.+]] = tosa.reciprocal [[LOG2]] : (tensor<f64>) -> tensor<f64>
  // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[LOG_ADD]], [[RECIP_LOG2]], [[SHIFT]] : (tensor<f64>, tensor<f64>, tensor<1xi8>) -> tensor<f64>
  // CHECK-NEXT: return [[MUL]] : tensor<f64>
  %0 = calc.logaddexp2 %arg0, %arg1 : (tensor<f64>, tensor<f64>) -> tensor<f64>
  return %0 : tensor<f64>
}

// CHECK-LABEL: func.func @test_logaddexp2_1d
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<4xf64>, [[ARG1:%.+]]: tensor<4xf64>) -> tensor<4xf64>
func.func @test_logaddexp2_1d(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<4xf64>}> : () -> tensor<4xf64>
  // CHECK-DAG: [[POW1:%.+]] = tosa.pow [[TWO]], [[ARG0]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[POW2:%.+]] = tosa.pow [[TWO]], [[ARG1]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[ADD:%.+]] = tosa.add [[POW1]], [[POW2]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[LOG_ADD:%.+]] = tosa.log [[ADD]] : (tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[LOG2:%.+]] = tosa.log [[TWO]] : (tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[RECIP_LOG2:%.+]] = tosa.reciprocal [[LOG2]] : (tensor<4xf64>) -> tensor<4xf64>
  // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[LOG_ADD]], [[RECIP_LOG2]], [[SHIFT]] : (tensor<4xf64>, tensor<4xf64>, tensor<1xi8>) -> tensor<4xf64>
  // CHECK-NEXT: return [[MUL]] : tensor<4xf64>
  %0 = calc.logaddexp2 %arg0, %arg1 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @test_logaddexp2_2d
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x3xf64>, [[ARG1:%.+]]: tensor<2x3xf64>) -> tensor<2x3xf64>
func.func @test_logaddexp2_2d(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // CHECK-DAG: [[TWO:%.+]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
  // CHECK-DAG: [[POW1:%.+]] = tosa.pow [[TWO]], [[ARG0]] : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[POW2:%.+]] = tosa.pow [[TWO]], [[ARG1]] : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[ADD:%.+]] = tosa.add [[POW1]], [[POW2]] : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[LOG_ADD:%.+]] = tosa.log [[ADD]] : (tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[LOG2:%.+]] = tosa.log [[TWO]] : (tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[RECIP_LOG2:%.+]] = tosa.reciprocal [[LOG2]] : (tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK-DAG: [[SHIFT:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: [[MUL:%.+]] = tosa.mul [[LOG_ADD]], [[RECIP_LOG2]], [[SHIFT]] : (tensor<2x3xf64>, tensor<2x3xf64>, tensor<1xi8>) -> tensor<2x3xf64>
  // CHECK-NEXT: return [[MUL]] : tensor<2x3xf64>
  %0 = calc.logaddexp2 %arg0, %arg1 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}