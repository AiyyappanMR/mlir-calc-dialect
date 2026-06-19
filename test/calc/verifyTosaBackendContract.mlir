 // RUN: not calc-opt --pass-pipeline="builtin.module(calc-to-arith, calc-verify-tosa-backend-contract)" %s 2>&1 | FileCheck %s
 // CHECK: error: Module does not conform to the TOSA backend contract.
 // CHECK-SAME: Found illegal 'calc' operation
 
// NOTE: This test intentionally triggers a pass failure.
// calc.mul has an arith lowering and will be converted successfully.
// calc.print has NO lowering — it survives calc-to-arith as-is,
// causing calc-verify-tosa-backend-contract to report an error.

module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    // calc.mul lowers cleanly to arith.muli — this op will be gone after calc-to-arith.
    %0 = calc.mul %arg0, %arg1 : i32

    // calc.print has no lowering — it survives calc-to-arith as-is,
    // and the verify pass will flag it as an illegal calc op.
    calc.print %0 : i32

    return %0 : i32
  }
}