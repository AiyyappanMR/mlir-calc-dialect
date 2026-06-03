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