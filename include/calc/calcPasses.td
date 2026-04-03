#ifndef CALC_PASSES
#define CALC_PASSES

include "mlir/Pass/PassBase.td"

def CalcToArithPass : Pass<"calc-to-arith", "::mlir::ModuleOp"> {
    let summary = "Lower calc dialect to arith dialect";
    let description = [{Lowers calc.add and calc.mul operations to arith dialect equivalents.}];
}

#endif 