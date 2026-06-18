#ifndef CALC_PASSES
#define CALC_PASSES

include "mlir/Pass/PassBase.td"

def CalcToArithPass : Pass<"calc-to-arith", "::mlir::ModuleOp"> {
    let summary = "Lower calc dialect to arith dialect";
    let description = [{Lowers calc.add and calc.mul operations to arith dialect equivalents.}];
}

def CalcToTosaPass : Pass<"calc-to-tosa", "::mlir::ModuleOp"> {
    let summary = "Lower calc dialect to tosa dialect";
    let description = [{Lowers calc.add and calc.mul operations to tosa dialect equivalents.}];
}

def CalcBroadcastPass : Pass<"calc-broadcast", "::mlir::ModuleOp"> {
    let summary = "Broadcast calc operations to have matching shapes";
    let description = [{
        Inserts reshape operations to ensure that all operands of calc 
        operations have matching ranks before lowering to TOSA.
    }];
}

def MemRefToLinalgPass : Pass<"memref-to-linalg", "::mlir::ModuleOp">{
    let summary = "Lowers memref dialect to linalg dialect";
    let description = [{
        Converts memref operations to their linalg equivalents to enable
        uniform lowering through the linalg pipeline.

        The primary conversion is `memref.copy`, which is rewritten as a
        `linalg.generic` op.
    }];
}

#endif 