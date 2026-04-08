// RUN: ./bin/calc-opt ../test/CalcToArith_test.mlir --calc-to-arith --mlir-print-ir-before-all


func.func @test_add_0(%arg0 : i32, %arg1 : i32)-> i32 {
    %result = calc.add %arg0, %arg1 : i32
    return %result : i32
}
func.func @test_add_1(%arg0 : f64, %arg1 : f64)-> f64 {
    %result = calc.add %arg0, %arg1 : f64
    return %result : f64
}
func.func @test_mul_0(%arg0 : i32, %arg1 : i32)-> i32 {
    %result = calc.mul %arg0, %arg1 : i32
    return %result : i32
}
func.func @test_mul_1(%arg0 : f64, %arg1 : f64)-> f64 {
    %result = calc.mul %arg0, %arg1 : f64
    return %result : f64
}
