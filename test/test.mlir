func.func @main() {
    %a = arith.constant 5 : i64
    %b = arith.constant 3 : i64
    %c = calc.add %a, %b : i64
    calc.print %c : i64
    return
}