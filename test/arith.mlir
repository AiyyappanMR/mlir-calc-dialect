// RUN: ./bin/calc-opt ../test/arith.mlir
// RUN(lowering to llvm): ./bin/calc-opt ../test/arith.mlir --pass-pipeline="builtin.module(func.func(convert-arith-to-llvm))"


func.func @main() {
    %a = arith.constant 5 : i64
    %b = arith.constant 3 : i64
    %c = arith.addi %a, %b : i64
    calc.print %c : i64
    return
}