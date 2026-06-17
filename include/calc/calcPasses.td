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

def CalcBufferizePass : Pass<"calc-bufferize", "::mlir::ModuleOp"> {
    let summary = "Bufferize calc operations";
    let description = [{
        Bufferizes calc operations to prepare for lowering to TOSA.
        This involves converting tensor types to memref types and inserting
        necessary load/store operations.
    }];
}

def CalcToMemRefPass : Pass<"calc-to-memref", "::mlir::ModuleOp"> {
  let summary = "Lower bufferized calc ops to memref ops";
  let description = [{
    Lowers bufferized calc ops (memref operands)
  }];
}

#endif 