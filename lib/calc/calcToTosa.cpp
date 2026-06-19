#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
class convertSplitOp : public mlir::OpRewritePattern<calc::splitOp> {
    using mlir::OpRewritePattern<calc::splitOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(calc::splitOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        int64_t rank = inputType.getRank();
        llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

        if (!inputType.hasStaticShape())
            return rewriter.notifyMatchFailure(op, "calc.split requires a statically-shaped input");

        llvm::ArrayRef<int64_t> splitSizes = op.getSplitSizes();

        // Resolve dim: default 0, normalize negative.
        int64_t dimVal = 0;
        if (mlir::IntegerAttr dimAttr = op.getDimAttr()) {
            dimVal = dimAttr.getValue().getSExtValue();
            if (dimVal < 0) dimVal += rank;
        }

        // Defensive bounds / consistency checks 
        if (rank <= 0)
            return rewriter.notifyMatchFailure(op, "calc.split requires rank > 0");
        if (dimVal < 0 || dimVal >= rank)
            return rewriter.notifyMatchFailure(op, "dim out of range");
        if (splitSizes.size() != op.getResults().size())
            return rewriter.notifyMatchFailure(op, "split_sizes/result count mismatch");


        mlir::tosa::shapeType shapeTy = mlir::tosa::shapeType::get(rewriter.getContext(), rank);
        llvm::SmallVector<mlir::Value> sliceResults;
        sliceResults.reserve(splitSizes.size());
        int64_t offset = 0;

        for (size_t i = 0; i < splitSizes.size(); ++i) {
            int64_t chunkSize = splitSizes[i];

            // start[dimVal] = running offset; all other dims start at 0.
            llvm::SmallVector<int64_t> startValues(rank, 0);
            startValues[dimVal] = offset;
            mlir::Value startShape = mlir::tosa::ConstShapeOp::create(rewriter, loc, shapeTy,
                rewriter.getIndexTensorAttr(llvm::ArrayRef<int64_t>(startValues)));

            // size[dimVal] = this chunk's size; all other dims retain the full input size.
            llvm::SmallVector<int64_t> sizeValues(inputShape.begin(), inputShape.end());
            sizeValues[dimVal] = chunkSize;
            mlir::Value sizeShape = mlir::tosa::ConstShapeOp::create(rewriter, loc, shapeTy,
                rewriter.getIndexTensorAttr(llvm::ArrayRef<int64_t>(sizeValues)));

            mlir::RankedTensorType sliceResultType = llvm::cast<mlir::RankedTensorType>(op.getResults()[i].getType());

            mlir::Value slice = mlir::tosa::SliceOp::create(rewriter, loc, sliceResultType, input, startShape, sizeShape);

            sliceResults.push_back(slice);
            offset += chunkSize;
        }

        rewriter.replaceOp(op, sliceResults);
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

namespace {
class convertProdOp : public mlir::OpRewritePattern<calc::prodOp> {
    using mlir::OpRewritePattern<calc::prodOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(calc::prodOp op, mlir::PatternRewriter &rewriter) const override {

        mlir::Location loc = op.getLoc();

        // Get the input tensor operand 
        mlir::Value input = op.getInput();
        mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        mlir::Type elemType = inputType.getElementType();

        // Get the Attributes
        mlir::Attribute dimAttr = op.getDimAttr();
        mlir::Attribute keepdimAttr = op.getKeepdimAttr();

        // Get the result type and rank
        mlir::RankedTensorType resultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType()); 
        int64_t resultRank = resultType.getRank();

        // If the input is scalar return the input itself 
        if (inputType.getRank() == 0) {
            rewriter.replaceOp(op, input);
            return mlir::success();
        }

        // If dim is specified
        if (dimAttr) {
            // Handling dim attribute
            int64_t dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
            if (dimVal < 0) dimVal += inputType.getRank();

            // ReduceProductOp always keeps the axis dim as size 1 (same rank as input).
            // Build an intermediate type with axis dim = 1, then reshape to the final resultType.
            auto interShape = inputType.getShape().vec();
            interShape[dimVal] = 1;
            mlir::RankedTensorType intermediateTy = mlir::RankedTensorType::get(interShape, elemType);

            // if keepdim=true, we can directly reduce to the resultType since ReduceProductOp will keep the dim with size 1.
            if (keepdimAttr && llvm::cast<mlir::BoolAttr>(keepdimAttr).getValue()) {
                mlir::Value result = mlir::tosa::ReduceProductOp::create(rewriter, loc, resultType, input, rewriter.getI32IntegerAttr(dimVal));
                rewriter.replaceOp(op, result);
                return mlir::success();
            } 

            else {
                // keepdim=false (explicit or default): reduce to intermediateTy, then reshape to resultType which has the axis dim removed.
                auto shapeTy = mlir::tosa::shapeType::get(rewriter.getContext(), resultRank);
                auto shapeAttr = rewriter.getIndexTensorAttr(resultType.getShape());
                mlir::Value shapeConst = mlir::tosa::ConstShapeOp::create(rewriter, loc, shapeTy, shapeAttr);
                mlir::Value out1 = mlir::tosa::ReduceProductOp::create(rewriter, loc, intermediateTy, input, rewriter.getI32IntegerAttr(dimVal));
                mlir::Value result = mlir::tosa::ReshapeOp::create(rewriter, loc, resultType, out1, shapeConst);
                rewriter.replaceOp(op, result);
                return mlir::success();
            }
        } else {

            // This lowering requires a compile-time element count in order to build the
            // intermediate flattened tensor type and reshape shape constant.
            if (!inputType.hasStaticShape()) {
                return rewriter.notifyMatchFailure(op,"calc.prod without dim requires a statically-shaped tensor input");
                    }

            // No dim: flatten to 1D, reduce → tensor<1xT>, then extract element and wrap as tensor<T>.
            int64_t inputElementCount = inputType.getNumElements();
            
            // create intermediate types for flattening and reduction
            mlir::RankedTensorType flatTy = mlir::RankedTensorType::get({inputElementCount}, elemType);
            mlir::RankedTensorType reduced1Ty = mlir::RankedTensorType::get({1}, elemType);
            auto shapeTy = mlir::tosa::shapeType::get(rewriter.getContext(), 1);
            auto shapeAttr = rewriter.getIndexTensorAttr(flatTy.getShape());

            // reshape the input tensor to 1D 
            mlir::Value shapeConst = mlir::tosa::ConstShapeOp::create(rewriter, loc, shapeTy, shapeAttr);
            mlir::Value reshapedInput = mlir::tosa::ReshapeOp::create(rewriter, loc, flatTy, input, shapeConst);

            // reduce the flattened tensor 
            mlir::Value reduced1 = mlir::tosa::ReduceProductOp::create(rewriter, loc, reduced1Ty, reshapedInput, rewriter.getI32IntegerAttr(0));

            // create a scalar tensor from the reduced result 
            mlir::Value idx0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
            mlir::Value scalar = mlir::tensor::ExtractOp::create(rewriter, loc, elemType, reduced1, mlir::ValueRange{idx0});
            mlir::Value result = mlir::tensor::FromElementsOp::create(rewriter, loc, resultType, mlir::ValueRange{scalar});

            // replace the original op with the final result
            rewriter.replaceOp(op, result);
            return mlir::success();
        }

    }
};
} // namespace

namespace { 
class convertCatMulAddOp : public mlir::OpRewritePattern<calc::catmuladdOp> {
  using mlir::OpRewritePattern<calc::catmuladdOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(calc::catmuladdOp op, mlir::PatternRewriter &rewriter) const override {

    // Get location
    mlir::Location loc = op.getLoc();

    // Initialize a dim value to 0 and if the dim attribute is present, update the dim value accordingly.
    int64_t dimVal = 0;
    mlir::Attribute dimAttr = op.getDimAttr();
    if (dimAttr) {
        dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    }

    // Get the result type
    mlir::RankedTensorType resultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    // Normalize negative dim value
    if (dimVal < 0) dimVal += rank;

    // defensive bounds check
    if (dimVal < 0 || dimVal >= rank)
        return rewriter.notifyMatchFailure(op, "dim out of range");

    mlir::OperandRange inputs1 = op.getInputs1();
    mlir::OperandRange inputs2 = op.getInputs2();

    std::vector<mlir::Value> Inputs1;
    std::vector<mlir::Value> Inputs2;

    for (mlir::Value v : inputs1) {
        Inputs1.push_back(v);
    }

    for (mlir::Value v : inputs2) {
        Inputs2.push_back(v);
    }

    mlir::Value cat1 = mlir::tosa::ConcatOp::create(rewriter, loc, resultType, Inputs1, rewriter.getI32IntegerAttr(dimVal));
    mlir::Value cat2 = mlir::tosa::ConcatOp::create(rewriter, loc, resultType, Inputs2, rewriter.getI32IntegerAttr(dimVal));

    // Create a constant shift value of 0 for the tosa::MulOp
    mlir::RankedTensorType shiftType =
    mlir::RankedTensorType::get({1}, rewriter.getI8Type()); // {1} not {}
    mlir::DenseElementsAttr shiftAttr =
    mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    mlir::Value shift =
    mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr);

    // Performs the element-wise multiplication of cat1 by cat2.
    mlir::Value catmuladdResult = mlir::tosa::MulOp::create(rewriter, loc, resultType, cat1, cat2, shift);

    mlir::Value scale = op.getScale();
    if (scale) {
        mlir::Value scaledResult = mlir::tosa::AddOp::create(rewriter, loc, resultType, catmuladdResult, scale);
        rewriter.replaceOp(op, scaledResult);
        return mlir::success();
    }

    // Replace the original catmuladd op with the final result.
    rewriter.replaceOp(op, catmuladdResult);
    return mlir::success();
  }
};
} // namespace



namespace { 
class convertSoftmaxOp : public mlir::OpRewritePattern<calc::softmaxOp> {
  using mlir::OpRewritePattern<calc::softmaxOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(calc::softmaxOp op, mlir::PatternRewriter &rewriter) const override {

    // Get location
    mlir::Location loc = op.getLoc();

    // Get the input operand and attribute
    mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    mlir::Attribute dimAttr = op.getDimAttr();

    int64_t dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();

    // normalize negative dim first
    if (dimVal < 0) dimVal += inputType.getRank();;

    // defensive bounds check
    if (dimVal < 0 || dimVal >= inputType.getRank())
        return rewriter.notifyMatchFailure(op, "dim out of range");

    // build a new type with dim axis set to 1
    llvm::SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[dimVal] = 1;
    mlir::RankedTensorType reducedType = mlir::RankedTensorType::get(reducedShape, inputType.getElementType());

    // Compute the exp(input) using tosa::ExpOp
    mlir::Value expInput = mlir::tosa::ExpOp::create(rewriter, loc, inputType, op.getInput());

    // Compute the sum of the exponentials along the specified dim using tosa::ReduceSumOp.
    mlir::Value sumExp = mlir::tosa::ReduceSumOp::create(rewriter, loc, reducedType, expInput,
                            rewriter.getI32IntegerAttr(dimVal));

    // Compute the reciprocal of the Exp sum using tosa::ReciprocalOp
    mlir::Value reciprocalSum = mlir::tosa::ReciprocalOp::create(rewriter, loc, reducedType, sumExp);

    // Create a constant shift value of 0 for the tosa::MulOp
    mlir::RankedTensorType shiftType = mlir::RankedTensorType::get({1}, rewriter.getI8Type()); // {1} not {}
    mlir::DenseElementsAttr shiftAttr = mlir::DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    mlir::Value shift = mlir::tosa::ConstOp::create(rewriter, op.getLoc(), shiftType, shiftAttr);

    // Multiply exp(input) with the reciprocal of the sum using tosa::MulOp to get the final softmax result.
    mlir::Value softmaxResult = mlir::tosa::MulOp::create(rewriter, loc, inputType, expInput, reciprocalSum, shift);

    // Replace the original softmax op with the final result.
    rewriter.replaceOp(op, softmaxResult);
    return mlir::success();
  }
};
} // namespace

namespace { 
class convertStackOp : public mlir::OpRewritePattern<calc::stackOp> {
  using mlir::OpRewritePattern<calc::stackOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(calc::stackOp op, mlir::PatternRewriter &rewriter) const override {

    // Get location
    mlir::Location loc = op.getLoc();

    // Initiaize a dim value to 0 and if the dim attribute is present, update the dim value accordingly.
    int64_t dimVal = 0;
    mlir::Attribute dimAttr = op.getDimAttr();
    if (dimAttr) {
        dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    }

    // Get the input operands and their types.
    mlir::OperandRange inputs = op.getInputs();
    mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(inputs[0].getType());
    std::vector<int64_t> inputShape = inputType.getShape().vec();

    // Get the result type and rank
    mlir::RankedTensorType resultType = llvm::cast<mlir::RankedTensorType>(op.getResult()[0].getType());
    int64_t resultRank = resultType.getRank();


    // normalize negative dim first
    if (dimVal < 0) dimVal += inputType.getRank()+1;

    // defensive bounds check
    if (dimVal < 0 || dimVal > inputType.getRank())
        return rewriter.notifyMatchFailure(op, "dim out of range");

    inputShape.insert(inputShape.begin() + dimVal, 1);

    // Pad the input tensor shapes with 1s at the dim axis so that they can be concatenated together. 
    // eg: if we have 3 input tensors of shape [2, 3] and dim=1, we will reshape them to [2, 1, 3] 
    //     and then concatenate along dim=1 to get a final result of shape [2, 3, 3].

    mlir::RankedTensorType reshapeType = mlir::RankedTensorType::get(inputShape, inputType.getElementType());
    mlir::tosa::shapeType reshapeTy = mlir::tosa::shapeType::get(rewriter.getContext(), resultRank);
    auto reshapeAttr = rewriter.getIndexTensorAttr(inputShape);
    mlir::Value shapeConst = mlir::tosa::ConstShapeOp::create(rewriter, loc, reshapeTy, reshapeAttr);

    // Initialize a vector to hold the reshaped input tensors.
    std::vector<mlir::Value> reshapedInputs;
    for (mlir::Value input : op.getInputs()) {

        // reshape each input tensor to the new shape with 1 at the dim axis using tosa::ReshapeOp.
        mlir::Value reshapedInput = mlir::tosa::ReshapeOp::create(rewriter, loc, reshapeType, input, shapeConst);
        reshapedInputs.push_back(reshapedInput);
    }

    // Concatenate the reshaped input tensors along the specified dim using tosa::ConcatOp to get the final result.
    mlir::Value stackResult = mlir::tosa::ConcatOp::create(rewriter, loc, resultType, reshapedInputs, rewriter.getI32IntegerAttr(dimVal));

    
    // Replace the original stack op with the final result.
    rewriter.replaceOp(op, stackResult);
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
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    target.addLegalDialect<mlir::func::FuncDialect, mlir::tosa::TosaDialect,
                           mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
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

    // Adding splitOp to the illegal ops.
    target.addIllegalOp<splitOp>();
    patterns.add<convertSplitOp>(&getContext());

    // Adding minimumOp to the illegal ops.
    target.addIllegalOp<minimumOp>();
    patterns.add<convertMinimumOp>(&getContext());

    // Adding addcmulOp to the illegal ops.
    target.addIllegalOp<addcmulOp>();
    patterns.add<convertAddcmulOp>(&getContext());

    // Adding entrOp to the illegal ops.
    target.addIllegalOp<entrOp>();
    patterns.add<convertEntrOp>(&getContext());

    // Adding prodOp to the illegal ops.
    target.addIllegalOp<prodOp>();
    patterns.add<convertProdOp>(&getContext());

    // Adding catmuladdOp to the illegal ops.
    target.addIllegalOp<catmuladdOp>();
    patterns.add<convertCatMulAddOp>(&getContext());

    // Adding softmaxOp to the illegal ops.
    target.addIllegalOp<softmaxOp>();
    patterns.add<convertSoftmaxOp>(&getContext());

    // Adding stackOp to the illegal ops.
    target.addIllegalOp<stackOp>();
    patterns.add<convertStackOp>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCalcToTosaPass() {
  return std::make_unique<CalcToTosaPass>();
}

} // namespace calc
