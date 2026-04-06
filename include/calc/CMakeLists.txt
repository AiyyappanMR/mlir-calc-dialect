add_mlir_dialect(calcOps calc)
set(LLVM_TARGET_DEFINITIONS calcPasses.td)
mlir_tablegen(calcPasses.h.inc -gen-pass-decls -name Calc)
add_public_tablegen_target(MLIRCalcPassesIncGen)