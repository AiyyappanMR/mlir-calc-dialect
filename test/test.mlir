// RUN: ./bin/calc-opt ../test/test.mlir

func.func @test_add_0(%arg0 : i32, %arg1 : i32)-> i32 {
    %result = calc.add %arg0, %arg1 : i32
    return %result : i32
}

func.func @test_add_1(%arg0 : f64, %arg1 : f64)-> f64 {
    %result = calc.add %arg0, %arg1 : f64
    return %result : f64
}

func.func @test_addt_0(%arg0 :tensor<4xf32>,%arg1 : tensor<4xf32>) -> tensor<4xf32>{
    %result = calc.addt %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
}

func.func @test_addt_1(%arg0 :tensor<4xi32>,%arg1 : tensor<4xi32>) -> tensor<4xi32>{
    %result = calc.addt %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
}

func.func @test_addt_2(%arg0 :tensor<3x3xi32>,%arg1 : tensor<3x3xi32>) -> tensor<3x3xi32>{
    %result = calc.addt %arg0, %arg1 : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
    return %result : tensor<3x3xi32>
}

func.func @test_addt_3(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
    %result = calc.addt %arg0, %arg1 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
}

func.func @test_addt_4(%arg0 : tensor<1xi32>, %arg1 : tensor<?xi32>) -> tensor<1xi32> {
    %result = calc.addt %arg0, %arg1 : (tensor<1xi32>, tensor<?xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
}

func.func @test_addt_5(%arg0 : tensor<1xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
    %result = calc.addt %arg0, %arg1 : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
}

// func.func @test_addt_4(%arg0 :tensor<3x1xf64>,%arg1 : tensor<3x3xf64>) -> tensor<3x3xf64>{
//     %result = calc.addt %arg0, %arg1 : (tensor<3x1xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
//     return %result : tensor<3x3xf64>
// }

