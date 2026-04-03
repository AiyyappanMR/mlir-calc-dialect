#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define GEN_PASS_DEF_CALCTOARITHPASS
#include "calc/calcPasses.h.inc"

namespace {
template<typename CalcOp, typename floatOp, typename intOp>
class convertElemWiseOp : public mlir::OpRewritePattern<CalcOp> {
    using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
        
        mlir::Type resultType = op.getResult().getType();
        
        auto lhs = op.getInput1();
        auto rhs = op.getInput2();

        if (mlir::isa<mlir::FloatType>(resultType)) {
            rewriter.replaceOpWithNewOp<floatOp>(op, resultType, lhs, rhs);
        } 
        else if (mlir::isa<mlir::IntegerType>(resultType)) {
            rewriter.replaceOpWithNewOp<intOp>(op, resultType, lhs, rhs);
        }
        else {
            return mlir::failure();
        }
        return mlir::success();
    }
};
}

namespace calc{
class CalcToArithPass : public impl::CalcToArithPassBase<CalcToArithPass> {
public:
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
    }

    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        //target.addIllegalDialect<calcDialect>();
        target.addLegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
        target.addIllegalOp<addOp, mulOp>();
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<convertElemWiseOp<addOp, mlir::arith::AddFOp, mlir::arith::AddIOp>>(&getContext());
        patterns.add<convertElemWiseOp<mulOp, mlir::arith::MulFOp, mlir::arith::MulIOp>>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
            return signalPassFailure();
    }
};

std::unique_ptr<mlir::Pass> createCalcToArithPass() {
    return std::make_unique<CalcToArithPass>();
}

} // namespace calc
