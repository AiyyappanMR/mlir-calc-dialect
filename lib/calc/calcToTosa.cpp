#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define GEN_PASS_DEF_CALCTOTOSAPASS
#include "calc/calcPasses.h.inc"

namespace { //class converter will be implemented here
template<typename CalcOp, typename tosaOp>
class convertElemWiseOp : public mlir::OpRewritePattern<CalcOp> {
    using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Value lhs = op.getInput1();
        mlir::Value rhs = op.getInput2();
        mlir::Type resultType = op.getResult().getType();
        
        mlir::RankedTensorType TensorType = mlir::RankedTensorType::get({},resultType);
        
        auto lhs_tensor = mlir::tensor::FromElementsOp::create(rewriter, op.getLoc(), TensorType, lhs);
        auto rhs_tensor = mlir::tensor::FromElementsOp::create(rewriter, op.getLoc(), TensorType, rhs);

        auto result = tosaOp::create(rewriter, op.getLoc(), TensorType, lhs_tensor, rhs_tensor);
        auto extracted = mlir::tensor::ExtractOp::create(rewriter, op.getLoc(), result, {});
        
        rewriter.replaceOp(op, extracted);
        return mlir::success();
    }
};
} //namespace

namespace{
template<typename CalcOp, typename tosaOp>
class covertMulElementWiseOp : public mlir::OpRewritePattern<CalcOp> {
    using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Value lhs = op.getInput1();
        mlir::Value rhs = op.getInput2();
        mlir::Type resultType = op.getResult().getType();
        
        mlir::RankedTensorType TensorType = mlir::RankedTensorType::get({},resultType);
        auto rankedShiftType = mlir::RankedTensorType::get({}, rewriter.getI8Type());
        auto unrankedShiftType = mlir::UnrankedTensorType::get(rewriter.getI8Type());

        auto lhs_tensor = mlir::tensor::FromElementsOp::create(rewriter, op.getLoc(), TensorType,lhs);
        auto rhs_tensor = mlir::tensor::FromElementsOp::create(rewriter, op.getLoc(), TensorType,rhs);

        auto shiftAttr = mlir::DenseElementsAttr::get(rankedShiftType, rewriter.getI8IntegerAttr(0));
        auto shift = mlir::tosa::ConstOp::create(rewriter, op.getLoc(), unrankedShiftType, shiftAttr);

        auto result = tosaOp::create(rewriter, op.getLoc(), TensorType, lhs_tensor, rhs_tensor, shift);
        auto extracted = mlir::tensor::ExtractOp::create(rewriter, op.getLoc(), result, {});
        
        rewriter.replaceOp(op, extracted);
        return mlir::success();
    }
};
} //namespace

namespace calc{
class CalcToTosaPass : public impl::CalcToTosaPassBase<CalcToTosaPass> {
public:
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::tosa::TosaDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
    }

    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        mlir::RewritePatternSet patterns(&getContext());
        target.addLegalDialect<mlir::func::FuncDialect, mlir::tosa::TosaDialect,mlir::tensor::TensorDialect>();
        target.addIllegalOp<addOp>();
        patterns.add<convertElemWiseOp<addOp, mlir::tosa::AddOp>>(&getContext());
        target.addIllegalOp<mulOp>();
        patterns.add<covertMulElementWiseOp<mulOp, mlir::tosa::MulOp>>(&getContext());
        
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
            return signalPassFailure();
        }
    };

std::unique_ptr<mlir::Pass> createCalcToTosaPass() {
    return std::make_unique<CalcToTosaPass>();

}

} //namespace calc

