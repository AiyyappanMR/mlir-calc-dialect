#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define GEN_PASS_DEF_CALCBROADCASTPASS
#include "calc/calcPasses.h.inc"

mlir::LogicalResult computeReshapeOutput(llvm::ArrayRef<int64_t> higherRankShape, llvm::ArrayRef<int64_t> lowerRankShape, llvm::SmallVector<int64_t> &reshapeOutputShape) {
    int64_t higherRank = higherRankShape.size();
    int64_t lowerRank = lowerRankShape.size();
    int64_t rankDiff = higherRank - lowerRank;

    int64_t higherRankDim;
    int64_t lowerRankDim;

    // Iterate through the dimensions of the lower rank shape and compare them with the corresponding dimensions of the higher rank shape.
    for (int64_t i = lowerRank - 1; i >= 0; i--) {
        higherRankDim = higherRankShape[i + rankDiff];
        lowerRankDim = lowerRankShape[i];
        // If the dimensions are not compatible for broadcasting, return failure. The dimensions are compatible if they are 
        //equal, or if one of them is 1 or dynamic.
        if (higherRankDim != lowerRankDim && (higherRankDim != 1 && higherRankDim != mlir::ShapedType::kDynamic) && (lowerRankDim != 1 && lowerRankDim != mlir::ShapedType::kDynamic)) {
            return mlir::failure();
        }
        // if the lower rank dimension is 1 then the output dimension is 1, otherwise it is the same as the lower rank dimension.
        reshapeOutputShape[i + rankDiff] = lowerRankDim == 1 ? 1 : lowerRankDim;
    }

    return mlir::success();
}

mlir::LogicalResult EqualizeRanks(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value &input1, mlir::Value &input2) {
    mlir::ImplicitLocOpBuilder builder(loc, rewriter);

    // Get the input tensor types and shapes
    auto input1Ty = llvm::dyn_cast<mlir::RankedTensorType>(input1.getType());
    auto input2Ty = llvm::dyn_cast<mlir::RankedTensorType>(input2.getType());

    // get the ranks of the input tensors
    int64_t input1Rank = input1Ty.getRank();
    int64_t input2Rank = input2Ty.getRank();

    // Create a variable to hold the higher and lower rank tensor values.
    mlir::Value higherTensorValue, lowerTensorValue;

    // Find out which tensor has high and low rank.
    if (input1Rank > input2Rank) {
        higherTensorValue = input1;
        lowerTensorValue = input2;
    } else {
        higherTensorValue = input2;
        lowerTensorValue = input1;
    }

    // get the shapes of high and low rank tensors
    mlir::ArrayRef<int64_t> higherRankShape = llvm::cast<mlir::RankedTensorType>(higherTensorValue.getType()).getShape();
    mlir::ArrayRef<int64_t> lowerRankShape = llvm::cast<mlir::RankedTensorType>(lowerTensorValue.getType()).getShape();
    
    // Create a vector to hold the output shape of the reshape operation, which will be the same as the higher rank shape 
    // and has all the dims set to 1.
    llvm::SmallVector<int64_t> reshapeOutputShape(higherRankShape.size(), 1);

    // Compute the output shape of the reshape operation based on the broadcasting rules. If the shapes are not compatible for broadcasting, return failure.
    if (computeReshapeOutput(higherRankShape, lowerRankShape, reshapeOutputShape).failed()) {
        return mlir::failure();
    }

    // Create a new ranked tensor type for the output of the reshape operation, which has the same element type as the 
    // lower rank tensor and the shape of the computed rank tensor.
    auto reshapeInputType = llvm::cast<mlir::RankedTensorType>(lowerTensorValue.getType());
    auto reshapeOutputType = mlir::RankedTensorType::get(llvm::ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());

    // Build a tosa.const_shape value: !tosa.shape<N> holding the target dims as index elements
    auto shapeType = mlir::tosa::shapeType::get(builder.getContext(), reshapeOutputShape.size());
    auto shapeAttr = builder.getIndexTensorAttr(reshapeOutputShape);
    mlir::Value reshapeOutputShapeValue = mlir::tosa::ConstShapeOp::create(builder, shapeType, shapeAttr)->getResult(0);

    // ReshapeOp::create with ImplicitLocOpBuilder takes (builder, resultType, input, shape)
    auto reshapeLower = mlir::tosa::ReshapeOp::create(
        builder, reshapeOutputType, lowerTensorValue, reshapeOutputShapeValue);

    if (input1Rank > input2Rank) {
        input1 = higherTensorValue;
        input2 = reshapeLower.getResult();
    } else {
        input1 = reshapeLower.getResult();
        input2 = higherTensorValue;
    }

    return mlir::success();
}


// This function computes the broadcasted shape of two input tensors and checks if they are compatible for broadcasting. 
// If they are, it returns the broadcasted shape; otherwise, it returns a failure.
mlir::LogicalResult ConvertRank(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::RankedTensorType outputType, mlir::Value &lhs, mlir::Value &rhs) {
    
    // Get the input tensor types and shapes
    auto lhsTy = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());

    // Check if both inputs are ranked tensors
    if (!lhsTy || !rhsTy) {
        return rewriter.notifyMatchFailure(loc, "Input not ranked tensor");
    }

    // Get the ranks of the input tensors
    int64_t input1Rank = lhsTy.getRank();
    int64_t input2Rank = rhsTy.getRank();

    // If the ranks are already equal, no need to reshape
    if (input1Rank == input2Rank) {
        return rewriter.notifyMatchFailure(loc, "cannot rewrite as its already correct");
    }

    // Create a copy of the input tensors to modify
    mlir::Value newlhs = lhs;
    mlir::Value newrhs = rhs;

    
    // Use the EqualizeRanks function to reshape the lower-rank tensor to match the higher rank
    if (EqualizeRanks(rewriter, loc, newlhs, newrhs).failed()) {
        return rewriter.notifyMatchFailure(loc, "failed to reshape ranks");
    }

    // Check if the reshaped tensors are compatible with the output type's shape.
    if (outputType) {
        if (outputType.getRank() != llvm::cast<mlir::RankedTensorType>(newlhs.getType()).getRank() ||
            outputType.getRank() != llvm::cast<mlir::RankedTensorType>(newrhs.getType()).getRank())
            return rewriter.notifyMatchFailure(loc, "the reshaped type doesn't agrees with the ranked output type");
    }

    // Update the original input tensors with the new reshaped tensors
    lhs = newlhs;
    rhs = newrhs;

    return mlir::success();
}


namespace {
template <typename CalcOp>
// This pattern matches a CalcOp with two tensor operands of different ranks, and rewrites it to have the same operation 
// but with the lower-rank tensor reshaped to match the higher rank using broadcasting rules.
class BroadcastPattern : public mlir::OpRewritePattern<CalcOp> {
    using mlir::OpRewritePattern<CalcOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(CalcOp op, mlir::PatternRewriter &rewriter) const override {
        
        // get the two input tensors values.
        mlir::Value lhs = op.getInput1();
        mlir::Value rhs = op.getInput2();

        // Check output type is ranked tensor or not.
        mlir::RankedTensorType outputType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
        if (!outputType) {
            return rewriter.notifyMatchFailure(op, "result type is not a ranked tensor");
        }

        // lhs and rhs are passed by reference so ConvertRank can update them with the reshaped values
        if (ConvertRank(rewriter, op.getLoc(), outputType, lhs, rhs).failed()) {
            return mlir::failure();
        }

        // modify the original op with the new operands.
        rewriter.replaceOpWithNewOp<CalcOp>(op, outputType, lhs, rhs);
        return mlir::success();
    }
};
} // namespace

namespace calc {
class CalcBroadcastPass : public impl::CalcBroadcastPassBase<CalcBroadcastPass> {
public:
    // Register dependent dialects
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::tosa::TosaDialect>();
    }
    // The main entry point for the pass
    void runOnOperation() override {
        // Get the current operation (module) and apply the broadcast pattern to all relevant ops within it.
        auto module = getOperation();
        mlir::RewritePatternSet patterns(&getContext());
        mlir::MLIRContext *ctx = &getContext();

        patterns.add<BroadcastPattern<calc::minimumOp>>(ctx);

        (void)mlir::applyPatternsGreedily(module, std::move(patterns));
    }
};

// function to create the pass
std::unique_ptr<mlir::Pass> createCalcBroadcastPass() {
    return std::make_unique<CalcBroadcastPass>();
}

} // namespace calc
