#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define GEN_PASS_DEF_CALCTOTOSAPASS
#include "calc/calcPasses.h.inc"

namespace { // class converter will be implemented here
template <typename CalcOp, typename tosaOp>
class convertElemWiseOp : public mlir::OpRewritePattern<CalcOp> {
  using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value lhs = op.getInput1();
    mlir::Value rhs = op.getInput2();
    mlir::Type resultType = op.getResult().getType();

    mlir::RankedTensorType TensorType =
        mlir::RankedTensorType::get({}, resultType);

    auto lhs_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(), TensorType, lhs);
    auto rhs_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(), TensorType, rhs);

    auto result = tosaOp::create(rewriter, op.getLoc(), TensorType, lhs_tensor,
                                 rhs_tensor);
    auto extracted =
        mlir::tensor::ExtractOp::create(rewriter, op.getLoc(), result, {});

    rewriter.replaceOp(op, extracted);
    return mlir::success();
  }
};
} // namespace

namespace {
template <typename CalcOp, typename tosaOp>
class convertMulElementWiseOp : public mlir::OpRewritePattern<CalcOp> {
  using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value lhs = op.getInput1();
    mlir::Value rhs = op.getInput2();
    mlir::Type resultType = op.getResult().getType();

    mlir::RankedTensorType TensorType =
        mlir::RankedTensorType::get({}, resultType);
    auto rankedShiftType =
        mlir::RankedTensorType::get({}, rewriter.getI8Type());
    auto unrankedShiftType =
        mlir::UnrankedTensorType::get(rewriter.getI8Type());

    auto lhs_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(), TensorType, lhs);
    auto rhs_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(), TensorType, rhs);

    auto shiftAttr = mlir::DenseElementsAttr::get(rankedShiftType,
                                                  rewriter.getI8IntegerAttr(0));
    auto shift = mlir::tosa::ConstOp::create(rewriter, op.getLoc(),
                                             unrankedShiftType, shiftAttr);

    auto result = tosaOp::create(rewriter, op.getLoc(), TensorType, lhs_tensor,
                                 rhs_tensor, shift);
    auto extracted =
        mlir::tensor::ExtractOp::create(rewriter, op.getLoc(), result, {});

    rewriter.replaceOp(op, extracted);
    return mlir::success();
  }
};
} // namespace

namespace {
template <typename CalcOp, typename tosaOp>
class convertTensorWiseOp : public mlir::OpRewritePattern<CalcOp> {
    using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;
        mlir::LogicalResult matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
            // Get the tensor operands and the result type and shape from the addtOp
            mlir::Value lhs = op.getInput1();
            mlir::Value rhs = op.getInput2();
            mlir::Type resultType = op.getResult().getType();
            // replace the addtOp with tosa::AddOp
            rewriter.replaceOpWithNewOp<tosaOp>(op, resultType, lhs, rhs);
            return mlir::success();
        }
    };
} // namespace

namespace {
class convertAddcmulOp : public mlir::OpRewritePattern<calc::addcmulOp> {
  using mlir::OpRewritePattern<calc::addcmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(calc::addcmulOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get the tensor operands and the result type and shape from the addcmulOp
    mlir::Value input = op.getInput();
    mlir::Value tensor1 = op.getTensor1();
    mlir::Value tensor2 = op.getTensor2();
    mlir::ShapedType resultType =
        llvm::cast<mlir::ShapedType>(op.getResult().getType());

    // Create a constant shift value of 0 for the tosa::MulOp
    mlir::RankedTensorType shiftType =
        mlir::RankedTensorType::get({1}, rewriter.getI8Type());  // {1} not {}
    mlir::DenseElementsAttr shiftAttr =
        mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    mlir::Value shift =
        mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr);

    // Performs the element-wise multiplication of tensor1 by tensor2.
    mlir::Value tensor_out_1 = mlir::tosa::MulOp::create(
        rewriter, op.getLoc(), resultType, tensor1, tensor2, shift);

    // check if the optional value attribute is present and compute the final
    // result..
    mlir::Attribute valueAttr = op.getValueAttr();
    if (valueAttr) {
      // If the value attribute is present, we make it a constant tensor and
      // multiply it with tensor_out_1, then add the result to the input tensor.
      mlir::DenseElementsAttr valueDense =
          mlir::DenseElementsAttr::get(resultType, valueAttr);
      mlir::Value valueTensor = mlir::tosa::ConstOp::create(
          rewriter, op.getLoc(), resultType, valueDense);
      mlir::Value tensor_out_2 = mlir::tosa::MulOp::create(
          rewriter, op.getLoc(), resultType, tensor_out_1, valueTensor, shift);
      mlir::Value finalResult = mlir::tosa::AddOp::create(
          rewriter, op.getLoc(), resultType, input, tensor_out_2);
      rewriter.replaceOp(op, finalResult);
      return mlir::success();
    } else {
      // If the value attribute is not present, we directly add tensor_out_1 to
      // the input tensor.
      mlir::Value finalResult = mlir::tosa::AddOp::create(
          rewriter, op.getLoc(), resultType, input, tensor_out_1);
      rewriter.replaceOp(op, finalResult);
      return mlir::success();
    }
  }
};
} // namespace

namespace calc {
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
    target.addLegalDialect<mlir::func::FuncDialect, mlir::tosa::TosaDialect,
                           mlir::tensor::TensorDialect>();
    target.addIllegalOp<addOp>();
    patterns.add<convertElemWiseOp<addOp, mlir::tosa::AddOp>>(&getContext());
    target.addIllegalOp<mulOp>();
    patterns.add<convertMulElementWiseOp<mulOp, mlir::tosa::MulOp>>(
        &getContext());
    // Adding addtOp to the illegal ops.
    target.addIllegalOp<addtOp>();
    patterns.add<convertTensorWiseOp<addtOp, mlir::tosa::AddOp>>(&getContext());

    // Adding addcmulOp to the illegal ops.
    target.addIllegalOp<addcmulOp>();
    patterns.add<convertAddcmulOp>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCalcToTosaPass() {
  return std::make_unique<CalcToTosaPass>();
}

} // namespace calc
