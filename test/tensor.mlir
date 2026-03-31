// RUN: ./bin/calc-opt ../test/tensor.mlir 
// RUN(lower to linalg): ./bin/calc-opt ../test/tensor.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" 

func.func @test_add(%arg0: tensor<4xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
}