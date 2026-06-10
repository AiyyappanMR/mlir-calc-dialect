#include "calc/calcOps.h"
#include "calc/calcDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace calc;

#define GET_OP_CLASSES
#include "calc/calcOps.cpp.inc"

// #define GET_PATTERN_CLASSES
// #include "calc/calcPatterns.cpp.inc" 

// Pattern to fuse addt and mult into addcmul.
// %0 = calc.mult %a, %b
// %1 = calc.addt %c, %0
// Replaces with:
// %1 = calc.addcmul %c, %a, %b



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

        //old cpp API for addcmul with Attribute
        // rewriter.replaceOpWithNewOp<calc::addcmulOp>(Op, Op.getType(), lhs, mullhs, mulrhs, mlir::Attribute{});

        // Here we use the cpp API for addcmul without specifying Attribute.
        rewriter.replaceOpWithNewOp<calc::addcmulOp>(Op, Op.getType(), lhs, mullhs, mulrhs);
        return success();
    }
};

static mlir::LogicalResult verifyResultShape(mlir::ShapedType inputType, mlir::ShapedType resultType, mlir::Attribute dimAttr, mlir::Attribute keepdimAttr) {
    // If dim is not specified, the result should be a scalar (rank 0 tensor).
    if (!dimAttr) {
        if (resultType.getRank() != 0) {
            return mlir::failure();
        }
        return mlir::success();
    }

    // If dim is specified, check the consistency of the result shape with the input shape, dim and keepdim attributes.
    int64_t dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    int64_t inputRank = inputType.getRank();
    
    // Handle negative dim value by converting it to the corresponding positive value.
    if (dimVal < 0) {
        dimVal += inputRank;
    }
    // get input and result shapes
    auto inputShape = inputType.getShape();
    auto resultShape = resultType.getShape();

    // if keepdim is present and true
    if (keepdimAttr && llvm::cast<mlir::BoolAttr>(keepdimAttr).getValue()) {
        // output rank should be the same as input rank
        if (resultType.getRank() != inputRank) {
            return mlir::failure();
        }

        // all dimensions except the dim should have the same size as input
        for (int64_t i = 0; i < inputRank; i++) {
            if (i == dimVal) {
                if (resultShape[i] != 1) {
                    return mlir::failure();
                }
            }
            else {
                if (resultShape[i] != inputShape[i]) {
                    return mlir::failure();
                }
            }
        }
    }
    else{
        // keepdim is false 
        // output rank should be input rank - 1
        if (resultType.getRank() != inputRank - 1) {
            return mlir::failure();
        }

        // all dimensions before dim should have the same size as input, and all dimensions after dim should have the same size 
        // as input but shifted by one position in the result.
        for (int64_t i = 0; i < inputRank; i++) {
            if (i < dimVal) {
                if (resultShape[i] != inputShape[i]) {
                    return mlir::failure();
                }
            }
            else if (i > dimVal) {
                if (resultShape[i - 1] != inputShape[i]) {
                    return mlir::failure();
                }
            }
        }
    }

    return mlir::success();
}

// calc.prod Op's Verifier method
mlir::LogicalResult calc::prodOp::verify() {
    
    //get attributes
    mlir::Attribute dim = getDimAttr();
    mlir::Attribute keepdim = getKeepdimAttr();

    // Check 1: keepdim without dim
    if (!dim && keepdim)
        return emitOpError("keepdim cannot be used without dim");

    // Check 2: dim range check
    if (dim) {
        mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(getInput().getType());
        int64_t rank = inputType.getRank();
        int64_t dimVal = llvm::cast<mlir::IntegerAttr>(dim).getValue().getSExtValue();

        if (dimVal < -rank || dimVal >= rank)
            return emitOpError("dim must be in range [-rank, rank-1]");
    }

    // Check 3: result shape verification based on input shape, dim and keepdim attributes.
    if (verifyResultShape(llvm::cast<mlir::ShapedType>(getInput().getType()), llvm::cast<mlir::ShapedType>(getResult().getType()), dim, keepdim).failed()){
        return emitOpError("result shape is not consistent with input shape, dim and keepdim attributes");
    }
        

    return mlir::success();
}

mlir::LogicalResult calc::catmuladdOp::verify() {

    // Get operand groups and result type
    mlir::OperandRange inputs1 = getInputs1();
    mlir::OperandRange inputs2 = getInputs2();
    mlir::RankedTensorType resultType = llvm::cast<mlir::RankedTensorType>(getResult().getType());
    int64_t resultRank = resultType.getRank();
    
    // Check: result must have rank >= 1 since we are concatenating along a dimension. If result is rank 0, it cannot have any dimensions to concatenate along.
    if (resultRank == 0)
        return emitOpError("requires rank >= 1 (cannot concatenate rank-0 tensors)");
    
        std::vector<int64_t> resultShape = resultType.getShape().vec(); // get result shape as vector

    // Check 0: each group must have at least one operand.
    if (inputs1.empty() || inputs2.empty())
        return emitOpError("requires at least one operand in each variadic group");

    // Check 1: dim range check
    int64_t dimVal = 0;
    mlir::Attribute dimAttr = getDimAttr();
    if (dimAttr) {
        dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
        if (dimVal < -resultRank || dimVal >= resultRank)
            return emitOpError("dim must be in range [-rank, rank-1]");
    }

    // Handle negative dim value by converting it to the corresponding positive value.
    if (dimVal < 0) dimVal += resultRank;

    // Check 2: all operands in inputs1 must have the same rank as the result and match the result shape on 
    // all non-concat dimensions

    // Initialize a variable to keep track of the total size along the concat dimension for inputs1
    int64_t input1DimSize = 0;
    for (mlir::Value operand : inputs1) {
        mlir::RankedTensorType operandType = llvm::cast<mlir::RankedTensorType>(operand.getType());
        std::vector<int64_t> operandShape = operandType.getShape().vec();
        // Check if the operand has the same rank as the result
        if ((int64_t)operandShape.size() != resultRank)
            return emitOpError("all operands in inputs1 must have the same rank as the result");
        for (int64_t i = 0; i < (int64_t)operandShape.size(); i++) {
            // For the concat dimension, we accumulate the size to compare with the result shape later.
            if (i == dimVal) {
                if (operandType.isDynamicDim(dimVal))
                    return emitOpError("requires static operand sizes along the concat dimension");
                input1DimSize += operandShape[i];
            }
            // For non-concat dimensions, the operand shape must match the result shape.
            else if (operandShape[i] != resultShape[i])
                return emitOpError("operands in inputs1 must match the result shape on all non-concat dimensions");
        }
    }

    // Check 3: all operands in inputs2 must have the same rank as the result and match the result shape on 
    // all non-concat dimensions

    // Initialize a variable to keep track of the total size along the concat dimension for inputs2
    int64_t input2DimSize = 0;
    for (mlir::Value operand : inputs2) {
        mlir::RankedTensorType operandType = llvm::cast<mlir::RankedTensorType>(operand.getType());
        std::vector<int64_t> operandShape = operandType.getShape().vec();
        // Check if the operand has the same rank as the result
        if ((int64_t)operandShape.size() != resultRank)
            return emitOpError("all operands in inputs2 must have the same rank as the result");
        for (int64_t i = 0; i < (int64_t)operandShape.size(); i++) {
            // For the concat dimension, we accumulate the size to compare with the result shape later.
            if (i == dimVal) {
                if (operandType.isDynamicDim(dimVal))
                    return emitOpError("requires static operand sizes along the concat dimension");
                input2DimSize += operandShape[i];
            }
            // For non-concat dimensions, the operand shape must match the result shape.
            else if (operandShape[i] != resultShape[i])
                return emitOpError("operands in inputs2 must match the result shape on all non-concat dimensions");
        }
    }

    // Check 4: inputs1 and inputs2 must produce the same concat-dim size since cat_a and cat_b are 
    // multiplied element-wise. Result shape along dim must equal that common size.
    if (input1DimSize != input2DimSize)
        return emitOpError("inputs1 and inputs2 must have the same total size along the concat dimension");
    if (input1DimSize != resultShape[dimVal])
        return emitOpError("the concat-dim size of each group must equal the result size along that dimension");

    // Check 5: optional scale operand must be broadcastable to the result shape
    mlir::Value scale = getScale();
    if (scale) {
        mlir::RankedTensorType scaleType = llvm::cast<mlir::RankedTensorType>(scale.getType());
        std::vector<int64_t> scaleShape = scaleType.getShape().vec();
        // Scale must have the same rank as the result and each dimension must be either 1 or match the 
        // corresponding dimension of the result.
        if ((int64_t)scaleShape.size() != resultRank)
            return emitOpError("scale operand must have the same rank as the result");
        for (int64_t i = 0; i < (int64_t)scaleShape.size(); i++) {
            if (scaleShape[i] != 1 && scaleShape[i] != resultShape[i])
                return emitOpError("scale operand must be broadcastable to the result shape");
        }
    }

    return mlir::success();
}

mlir::LogicalResult calc::softmaxOp::verify() {
    
    // Get input rank and dim attribute value
    mlir::RankedTensorType inputType = llvm::cast<mlir::RankedTensorType>(getInput().getType());
    int64_t rank = inputType.getRank();

    mlir::Attribute dim = getDimAttr();
    int64_t dimVal = llvm::cast<mlir::IntegerAttr>(dim).getValue().getSExtValue();

    // Check 1: dim range check
     if (dimVal < -rank || dimVal >= rank){
            return emitOpError("dim must be in range [-rank, rank-1]");
    }
    return mlir::success();
}

// Canonicalization pattern to fuse addt and mult into addcmul.
void calc::addtOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<FuseAddMul>(context);
  // populateWithGenerated(results);
}