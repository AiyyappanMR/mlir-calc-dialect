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

mlir::LogicalResult ToFloat(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value &input, mlir::Type targetFloatType) {
    // Check if the input tensor is already of float type, if so, no need to cast.
    mlir::RankedTensorType inputTensorType = llvm::cast<mlir::RankedTensorType>(input.getType());
    if (llvm::isa<mlir::FloatType>(inputTensorType.getElementType())) {
        return mlir::success();
    }
    mlir::RankedTensorType castResultType = mlir::RankedTensorType::get(inputTensorType.getShape(), targetFloatType);
    input = mlir::tosa::CastOp::create(rewriter, loc, castResultType, input);
    return mlir::success();
}

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

namespace {
class convertLogaddexp2Op : public mlir::OpRewritePattern<calc::logaddexp2Op> {
    using mlir::OpRewritePattern<calc::logaddexp2Op>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(calc::logaddexp2Op op,
    mlir::PatternRewriter &rewriter) const override {
    // Get the tensor operands and the result type and shape from the logaddexp2Op
    mlir::Value input1 = op.getInput1();
    mlir::Value input2 = op.getInput2();
    mlir::ShapedType resultType =
    llvm::cast<mlir::ShapedType>(op.getResult().getType());

    // Create a constant shift value of 0 for the tosa::MulOp
    mlir::RankedTensorType shiftType =
    mlir::RankedTensorType::get({1}, rewriter.getI8Type()); // {1} not {}
    mlir::DenseElementsAttr shiftAttr =
    mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    mlir::Value shift =
    mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr); 
    // Create a constant tensor with the same shape as the input tensors and all values set to 
    // 2.0 for the tosa::PowOp and tosa::LogOp.
    mlir::DenseElementsAttr twoTensorAttr = 
    mlir::DenseElementsAttr::get(resultType, rewriter.getF64FloatAttr(2.0));
    mlir::Value twoTensor =
    mlir::tosa::ConstOp::create(rewriter, op.getLoc(), resultType, twoTensorAttr);

    // compute 2^input1 and 2^input2 using tosa::PowOp, then add the results together.
    mlir::Value pow_input1 = mlir::tosa::PowOp::create(
    rewriter, op.getLoc(), resultType, twoTensor, input1);
    mlir::Value pow_input2 = mlir::tosa::PowOp::create(
    rewriter, op.getLoc(), resultType, twoTensor, input2);
    mlir::Value add_pow = mlir::tosa::AddOp::create(
    rewriter, op.getLoc(), resultType, pow_input1, pow_input2);

    // compute log(2^input1 + 2^input2) and log(2) using tosa::LogOp, because log base 2 is not supported in tosa. 
    mlir::Value log_add_pow_val = mlir::tosa::LogOp::create(
    rewriter, op.getLoc(), resultType, add_pow);
    mlir::Value log2_val = mlir::tosa::LogOp::create(
    rewriter, op.getLoc(), resultType, twoTensor);

    // Also in tosa, there is no div op specifically, so we compute the reciprocal of log(2) and 
    // then multiply it with log(2^input1 + 2^input2) to get the final result.
    // Result = log(2^input1 + 2^input2) / log(2) = log(2^input1 + 2^input2) * (1/log(2))
    mlir::Value recip = mlir::tosa::ReciprocalOp::create(
    rewriter, op.getLoc(), resultType, log2_val);
    mlir::Value result = mlir::tosa::MulOp::create(
    rewriter, op.getLoc(), resultType, log_add_pow_val, recip, shift);
    rewriter.replaceOp(op, result);
    return mlir::success();
    }
};
} // namespace

namespace {
class convertMinimumOp : public mlir::OpRewritePattern<calc::minimumOp> {
    using mlir::OpRewritePattern<calc::minimumOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(calc::minimumOp op, mlir::PatternRewriter &rewriter) const override {
        // Get the tensor operands and the result type and shape from the logaddexp2Op
        mlir::Value input1 = op.getInput1();
        mlir::Value input2 = op.getInput2();
        mlir::ShapedType resultType =
        llvm::cast<mlir::ShapedType>(op.getResult().getType());
        mlir::Type elementType = resultType.getElementType();

        // Minimum(a, b) = ((a + b) - abs(a - b)) / 2
        mlir::Value out1 = mlir::tosa::AddOp::create(
        rewriter, op.getLoc(), resultType, input1, input2);
        mlir::Value out2 = mlir::tosa::SubOp::create(
        rewriter, op.getLoc(), resultType, input1, input2);
        mlir::Value abs_out = mlir::tosa::AbsOp::create(
        rewriter, op.getLoc(), resultType, out2);
        mlir::Value sub_out = mlir::tosa::SubOp::create(
        rewriter, op.getLoc(), resultType, out1, abs_out);

        // Create a constant tensor with the same shape as the input tensors and all values set to 2 or 2.0 for the tosa::ReciprocalOp.
        // Check if the element type is integer or float to determine the value of the constant tensor.
        mlir::DenseElementsAttr twoTensorAttr;
        if (elementType.isInteger(32)){
            twoTensorAttr = mlir::DenseElementsAttr::get(resultType, rewriter.getI32IntegerAttr(2));
        }
        else if (elementType.isInteger(64)){
            twoTensorAttr = mlir::DenseElementsAttr::get(resultType, rewriter.getI64IntegerAttr(2));
        }
        else if (elementType.isF32()){
            twoTensorAttr = mlir::DenseElementsAttr::get(resultType, rewriter.getF32FloatAttr(2.0));
        }
        else if (elementType.isF64()){
            twoTensorAttr = mlir::DenseElementsAttr::get(resultType, rewriter.getF64FloatAttr(2.0));
        }
        else {
            return rewriter.notifyMatchFailure(op, "unsupported element type");
        }
        mlir::Value twoTensor = mlir::tosa::ConstOp::create(rewriter, op.getLoc(), resultType, twoTensorAttr);

        // Create a constant shift value of 0 for the tosa::MulOp
        mlir::RankedTensorType shiftType =
        mlir::RankedTensorType::get({1}, rewriter.getI8Type()); // {1} not {}
        mlir::DenseElementsAttr shiftAttr =
        mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
        mlir::Value shift =
        mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr);
        
        // calculate the reciprocal of 2 and then multiply it with the sub_out to get the final result.
        mlir::Value result;
        if (llvm::isa<mlir::IntegerType>(elementType)) {
            // For integer types, we need to use the tosa::DivOp instead of multiplying by the reciprocal, because the reciprocal of 2 would be 0 in integer division.
            result = mlir::tosa::IntDivOp::create(
            rewriter, op.getLoc(), resultType, sub_out, twoTensor);
        }
        else if (llvm::isa<mlir::FloatType>(elementType)) {
            // For float types, we can directly multiply by the reciprocal of 2 to get the final result.
            mlir::Value recip = mlir::tosa::ReciprocalOp::create(
            rewriter, op.getLoc(), resultType, twoTensor);
            result = mlir::tosa::MulOp::create(
            rewriter, op.getLoc(), resultType, sub_out, recip, shift);
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};
} // namespace


namespace {
class convertEntrOp : public mlir::OpRewritePattern<calc::entrOp> {
    using mlir::OpRewritePattern<calc::entrOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(calc::entrOp op, mlir::PatternRewriter &rewriter) const override {

        mlir::Location loc = op.getLoc();

        // Get the tensor operand 
        mlir::Value input = op.getInput();
        mlir::ShapedType inputType = llvm::cast<mlir::ShapedType>(input.getType());
        mlir::Type elemType = inputType.getElementType();

        // Get the result type
        mlir::Value result = op.getResult();
        mlir::ShapedType resultType = llvm::cast<mlir::ShapedType>(result.getType());
        mlir::Type resultElemType = resultType.getElementType();

        // If the element type of the input tensor is not a float type, we need to convert it to a float type before 
        // lowering to tosa.
        if (!elemType.isF32() && !elemType.isF64()) {
            if ((ToFloat(rewriter, op.getLoc(), input, resultElemType)).failed()) {
                return rewriter.notifyMatchFailure(op, "failed to convert integer to float");
            }
            // After conversion, we need to update the inputType and elemType.
            inputType = llvm::cast<mlir::ShapedType>(input.getType());
            elemType = inputType.getElementType();
        }

        // Get masks for input == 0, and input < 0
        mlir::DenseElementsAttr zeroTensorAttr = mlir::DenseElementsAttr::get(inputType, rewriter.getFloatAttr(elemType, 0.0));
        mlir::Value zeroTensor = mlir::tosa::ConstOp::create(rewriter, op.getLoc(), inputType, zeroTensorAttr);
        
        mlir::RankedTensorType MaskType = mlir::RankedTensorType::get(inputType.getShape(), rewriter.getI1Type());
        mlir::Value eqMask = mlir::tosa::EqualOp::create(rewriter, op.getLoc(), MaskType, input, zeroTensor);
        mlir::Value ltMask = mlir::tosa::GreaterOp::create(rewriter, loc, MaskType, zeroTensor, input);

        // Create a constant shift value of 0 for the tosa::MulOp
        mlir::RankedTensorType shiftType =
        mlir::RankedTensorType::get({1}, rewriter.getI8Type()); // {1} not {}
        mlir::DenseElementsAttr shiftAttr =
        mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
        mlir::Value shift =
        mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr);

        // Compute the entropy using the formula: entr(x) = -x * log(x) for whole tensor.
        mlir::Value logVal = mlir::tosa::LogOp::create(rewriter, loc, resultType, input);
        mlir::Value mulVal = mlir::tosa::MulOp::create(rewriter, loc, resultType, input, logVal, shift);
        mlir::Value entrPositive = mlir::tosa::NegateOp::create(rewriter, loc, resultType, mulVal);

        // Create a constant tensor with the same shape as the input tensor and all values set to negative infinity.
        mlir::DenseElementsAttr negInfAttr = mlir::DenseElementsAttr::get(resultType, rewriter.getFloatAttr(resultElemType, -std::numeric_limits<double>::infinity()));
        mlir::Value negInfTensor = mlir::tosa::ConstOp::create(rewriter, loc, resultType, negInfAttr);

        // first select op to select between entrPositive and zeroTensor based on eqMask.
        // eg : [0.5, -1.0, 0.0, 2.0] -> [entr(0.5), entr(-1.0), entr(0.0), entr(2.0)] 
        // eqMask : [false, false, true, false] 
        // after first select : [entr(0.5), entr(-1.0), 0.0, entr(2.0)]
        mlir::Value out1 = mlir::tosa::SelectOp::create(rewriter, loc, resultType, eqMask, zeroTensor, entrPositive);

        // second select op to select between entrPositive and negative infinity based on ltMask.
        // eg : [entr(0.5), entr(-1.0), 0.0, entr(2.0)]
        // ltMask : [false, true, false, false]
        // final output : [entr(0.5), -inf, 0.0, entr(2.0)]
        mlir::Value out2 = mlir::tosa::SelectOp::create(rewriter, loc, resultType, ltMask, negInfTensor, out1);

        rewriter.replaceOp(op, out2);
        return mlir::success();
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

    // Adding logaddexp2Op to the illegal ops.
    target.addIllegalOp<logaddexp2Op>();
    patterns.add<convertLogaddexp2Op>(&getContext());

    // Adding minimumOp to the illegal ops.
    target.addIllegalOp<minimumOp>();
    patterns.add<convertMinimumOp>(&getContext());

    // Adding addcmulOp to the illegal ops.
    target.addIllegalOp<addcmulOp>();
    patterns.add<convertAddcmulOp>(&getContext());

    // Adding entrOp to the illegal ops.
    target.addIllegalOp<entrOp>();
    patterns.add<convertEntrOp>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCalcToTosaPass() {
  return std::make_unique<CalcToTosaPass>();
}

} // namespace calc
