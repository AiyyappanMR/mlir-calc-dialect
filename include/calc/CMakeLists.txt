add_mlir_dialect(calcOps calc)
set(LLVM_TARGET_DEFINITIONS calcPasses.td)
mlir_tablegen(calcPasses.h.inc -gen-pass-decls -name Calc)

#generate the C-API header and implementation for the passes
mlir_tablegen(calcTransforms.capi.h.inc -gen-pass-capi-header --prefix Calc)
mlir_tablegen(calcTransforms.capi.cpp.inc -gen-pass-capi-impl --prefix Calc)
add_public_tablegen_target(MLIRCalcTransformsIncGen)