// Tests bufferization of the calc dialect (calc.stack).

// RUN: calc-opt --calc-bufferize --calc-to-memref %s | FileCheck %s

// Test 1: Standard 1D Tensor Stack along Dim 0
// CHECK-LABEL: func.func @test_stack_1d_dim0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi32>, [[ARG1:%.+]]: tensor<4xi32>) -> tensor<2x4xi32>
func.func @test_stack_1d_dim0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<2x4xi32> {
    // CHECK-DAG:  [[BUF1:%.+]] = bufferization.to_buffer [[ARG1]] : tensor<4xi32> to memref<4xi32>
    // CHECK-DAG:  [[BUF0:%.+]] = bufferization.to_buffer [[ARG0]] : tensor<4xi32> to memref<4xi32>
    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<2x4xi32>
    // CHECK:      [[RES_TENSOR:%.+]] = bufferization.to_tensor [[ALLOC]] restrict : memref<2x4xi32> to tensor<2x4xi32>
    // CHECK:      [[SUBVIEW0:%.+]] = memref.subview [[ALLOC]][0, 0] [1, 4] [1, 1] : memref<2x4xi32> to memref<1x4xi32, strided<[4, 1]>>
    // CHECK:      [[EXPAND0:%.+]] = memref.expand_shape [[BUF0]] {{\[\[}}0, 1]] output_shape [1, 4] : memref<4xi32> into memref<1x4xi32>
    // CHECK:      memref.copy [[EXPAND0]], [[SUBVIEW0]]
    // CHECK:      [[SUBVIEW1:%.+]] = memref.subview [[ALLOC]][1, 0] [1, 4] [1, 1] : memref<2x4xi32> to memref<1x4xi32, strided<[4, 1], offset: 4>>
    // CHECK:      [[EXPAND1:%.+]] = memref.expand_shape [[BUF1]] {{\[\[}}0, 1]] output_shape [1, 4] : memref<4xi32> into memref<1x4xi32>
    // CHECK:      memref.copy [[EXPAND1]], [[SUBVIEW1]]
    // CHECK:      return [[RES_TENSOR]] : tensor<2x4xi32>
    %0 = calc.stack %arg0, %arg1 {dim = 0 : si32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
}

// -----

// Test 2: Different Data Type (f32) to ensure type propagation
// CHECK-LABEL: func.func @test_stack_1d_f32
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>, [[ARG1:%.+]]: tensor<4xf32>) -> tensor<2x4xf32>
func.func @test_stack_1d_f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    // CHECK-DAG:  [[BUF1:%.+]] = bufferization.to_buffer [[ARG1]] : tensor<4xf32> to memref<4xf32>
    // CHECK-DAG:  [[BUF0:%.+]] = bufferization.to_buffer [[ARG0]] : tensor<4xf32> to memref<4xf32>
    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<2x4xf32>
    // CHECK:      [[RES_TENSOR:%.+]] = bufferization.to_tensor [[ALLOC]] restrict : memref<2x4xf32> to tensor<2x4xf32>
    // CHECK:      [[SUBVIEW0:%.+]] = memref.subview [[ALLOC]][0, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<1x4xf32, strided<[4, 1]>>
    // CHECK:      [[EXPAND0:%.+]] = memref.expand_shape [[BUF0]] {{\[\[}}0, 1]] output_shape [1, 4] : memref<4xf32> into memref<1x4xf32>
    // CHECK:      memref.copy [[EXPAND0]], [[SUBVIEW0]]
    // CHECK:      [[SUBVIEW1:%.+]] = memref.subview [[ALLOC]][1, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<1x4xf32, strided<[4, 1], offset: 4>>
    // CHECK:      [[EXPAND1:%.+]] = memref.expand_shape [[BUF1]] {{\[\[}}0, 1]] output_shape [1, 4] : memref<4xf32> into memref<1x4xf32>
    // CHECK:      memref.copy [[EXPAND1]], [[SUBVIEW1]]
    // CHECK:      return [[RES_TENSOR]] : tensor<2x4xf32>
    %0 = calc.stack %arg0, %arg1 {dim = 0 : si32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
}
// -----

// Test 3: Stacking along Dimension 1 (Changes subview offsets and strides)
// CHECK-LABEL: func.func @test_stack_1d_dim1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi32>, [[ARG1:%.+]]: tensor<4xi32>) -> tensor<4x2xi32>
func.func @test_stack_1d_dim1(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4x2xi32> {
    // CHECK-DAG:  [[BUF1:%.+]] = bufferization.to_buffer [[ARG1]] : tensor<4xi32> to memref<4xi32>
    // CHECK-DAG:  [[BUF0:%.+]] = bufferization.to_buffer [[ARG0]] : tensor<4xi32> to memref<4xi32>
    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<4x2xi32>
    // CHECK:      [[RES_TENSOR:%.+]] = bufferization.to_tensor [[ALLOC]] restrict : memref<4x2xi32> to tensor<4x2xi32>
    // CHECK:      [[SUBVIEW0:%.+]] = memref.subview [[ALLOC]][0, 0] [4, 1] [1, 1] : memref<4x2xi32> to memref<4x1xi32, strided<[2, 1]>>
    // CHECK:      [[EXPAND0:%.+]] = memref.expand_shape [[BUF0]] {{\[\[}}0, 1]] output_shape [4, 1] : memref<4xi32> into memref<4x1xi32>
    // CHECK:      memref.copy [[EXPAND0]], [[SUBVIEW0]]
    // CHECK:      [[SUBVIEW1:%.+]] = memref.subview [[ALLOC]][0, 1] [4, 1] [1, 1] : memref<4x2xi32> to memref<4x1xi32, strided<[2, 1], offset: 1>>
    // CHECK:      [[EXPAND1:%.+]] = memref.expand_shape [[BUF1]] {{\[\[}}0, 1]] output_shape [4, 1] : memref<4xi32> into memref<4x1xi32>
    // CHECK:      memref.copy [[EXPAND1]], [[SUBVIEW1]]
    // CHECK:      return [[RES_TENSOR]] : tensor<4x2xi32>
    %0 = calc.stack %arg0, %arg1 {dim = 1 : si32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4x2xi32>
    return %0 : tensor<4x2xi32>
}

// -----

// Test 4: Single element tensor stacking (Edge case size 1)
// CHECK-LABEL: func.func @test_stack_single_element
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1xi32>, [[ARG1:%.+]]: tensor<1xi32>) -> tensor<2x1xi32>
func.func @test_stack_single_element(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<2x1xi32> {
    // CHECK-DAG:  [[BUF1:%.+]] = bufferization.to_buffer [[ARG1]] : tensor<1xi32> to memref<1xi32>
    // CHECK-DAG:  [[BUF0:%.+]] = bufferization.to_buffer [[ARG0]] : tensor<1xi32> to memref<1xi32>
    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<2x1xi32>
    // CHECK:      [[RES_TENSOR:%.+]] = bufferization.to_tensor [[ALLOC]] restrict : memref<2x1xi32> to tensor<2x1xi32>
    // CHECK:      [[SUBVIEW0:%.+]] = memref.subview [[ALLOC]][0, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1x1xi32, strided<[1, 1]>>
    // CHECK:      [[EXPAND0:%.+]] = memref.expand_shape [[BUF0]] {{\[\[}}0, 1]] output_shape [1, 1] : memref<1xi32> into memref<1x1xi32>
    // CHECK:      memref.copy [[EXPAND0]], [[SUBVIEW0]]
    // CHECK:      [[SUBVIEW1:%.+]] = memref.subview [[ALLOC]][1, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1x1xi32, strided<[1, 1], offset: 1>>
    // CHECK:      [[EXPAND1:%.+]] = memref.expand_shape [[BUF1]] {{\[\[}}0, 1]] output_shape [1, 1] : memref<1xi32> into memref<1x1xi32>
    // CHECK:      memref.copy [[EXPAND1]], [[SUBVIEW1]]
    // CHECK:      return [[RES_TENSOR]] : tensor<2x1xi32>
    %0 = calc.stack %arg0, %arg1 {dim = 0 : si32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
}

// -----

// Test 5: 2D Tensor Stack to 3D Tensor (Higher dimensionality test)
// CHECK-LABEL: func.func @test_stack_2d_to_3d
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3xi32>, [[ARG1:%.+]]: tensor<2x3xi32>) -> tensor<2x2x3xi32>
func.func @test_stack_2d_to_3d(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x2x3xi32> {
    // CHECK-DAG:  [[BUF1:%.+]] = bufferization.to_buffer [[ARG1]] : tensor<2x3xi32> to memref<2x3xi32>
    // CHECK-DAG:  [[BUF0:%.+]] = bufferization.to_buffer [[ARG0]] : tensor<2x3xi32> to memref<2x3xi32>
    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<2x2x3xi32>
    // CHECK:      [[RES_TENSOR:%.+]] = bufferization.to_tensor [[ALLOC]] restrict : memref<2x2x3xi32> to tensor<2x2x3xi32>
    // CHECK:      [[SUBVIEW0:%.+]] = memref.subview [[ALLOC]][0, 0, 0] [1, 2, 3] [1, 1, 1] : memref<2x2x3xi32> to memref<1x2x3xi32, strided<[6, 3, 1]>>
    // CHECK:      [[EXPAND0:%.+]] = memref.expand_shape [[BUF0]] {{\[\[}}0, 1], [2]] output_shape [1, 2, 3] : memref<2x3xi32> into memref<1x2x3xi32>
    // CHECK:      memref.copy [[EXPAND0]], [[SUBVIEW0]]
    // CHECK:      [[SUBVIEW1:%.+]] = memref.subview [[ALLOC]][1, 0, 0] [1, 2, 3] [1, 1, 1] : memref<2x2x3xi32> to memref<1x2x3xi32, strided<[6, 3, 1], offset: 6>>
    // CHECK:      [[EXPAND1:%.+]] = memref.expand_shape [[BUF1]] {{\[\[}}0, 1], [2]] output_shape [1, 2, 3] : memref<2x3xi32> into memref<1x2x3xi32>
    // CHECK:      memref.copy [[EXPAND1]], [[SUBVIEW1]]
    // CHECK:      return [[RES_TENSOR]] : tensor<2x2x3xi32>
    %0 = calc.stack %arg0, %arg1 {dim = 0 : si32} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x2x3xi32>
    return %0 : tensor<2x2x3xi32>
}

// -----