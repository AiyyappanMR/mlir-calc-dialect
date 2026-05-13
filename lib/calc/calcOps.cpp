#include "calc/calcOps.h"
#include "calc/calcDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace calc;

#define GET_OP_CLASSES
#include "calc/calcOps.cpp.inc"

#define GET_PATTERN_CLASSES
#include "calc/calcPatterns.cpp.inc" 

// Pattern to fuse addt and mult into addcmul.
// %0 = calc.mult %a, %b
// %1 = calc.addt %c, %0
// Replaces with:
// %1 = calc.addcmul %c, %a, %b


// C++ pattern commented out — replaced by DRR pattern in calcPatterns.td
/*
struct FuseAddMul : public OpRewritePattern<calc::addtOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(calc::addtOp Op, PatternRewriter &rewriter) const override {
        // Get the operands of addt.
        mlir::Value lhs = Op.getOperand(0);
        mlir::Value rhs = Op.getOperand(1);

        // Check if rhs has been used by only one operation and that operation is a mult.
        if (!rhs.hasOneUse()) {
            return failure();
        }

        // Check if the rhs is defined by a mult operation.
        auto mul = rhs.getDefiningOp<calc::multOp>();
        if (!mul) {
            return failure();
        }

        // Get the operands of the mult operation.
        mlir::Value mullhs = mul.getOperand(0);
        mlir::Value mulrhs = mul.getOperand(1);

        // Replace the addt operation with an addcmul operation.
        rewriter.replaceOpWithNewOp<calc::addcmulOp>(Op, Op.getType(), lhs, mullhs, mulrhs, mlir::Attribute{});
        return success();
    }
};
*/

// Canonicalization pattern to fuse addt and mult into addcmul.
void calc::addtOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // results.add<FuseAddMul>(context);
  populateWithGenerated(results);
}