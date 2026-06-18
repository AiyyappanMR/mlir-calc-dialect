#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "calc/calcDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CALCVERIFYTOSABACKENDCONTRACTPASS
#include "calc/calcPasses.h.inc"

namespace calc {

namespace {
class CalcVerifyTosaBackendContractPass
    : public impl::CalcVerifyTosaBackendContractPassBase<CalcVerifyTosaBackendContractPass> {
  void runOnOperation() override {

    // This pass verifies that the IR module conforms to the TOSA backend contract —
    // i.e., all `calc` dialect ops have been fully lowered before reaching the
    // backend. It does NOT perform any lowering itself; it only asserts that none of the calc
    // ops remain in the IR module.
    
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(*context);

    // Explicitly blacklist the calc dialect: no calc ops are allowed.
    target.addIllegalDialect<calcDialect>();

    // Everything else (tosa, tensor, arith, func, etc.) is perfectly fine.
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

    mlir::RewritePatternSet patterns(context);

    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
      // We avoid `module.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      mlir::emitError(module.getLoc())
          << "Module does not conform to the TOSA backend contract. "
             "Found illegal 'calc' operation after lowering to TOSA.";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createCalcVerifyTosaBackendContractPass() {
  return std::make_unique<CalcVerifyTosaBackendContractPass>();
}

} // namespace calc
