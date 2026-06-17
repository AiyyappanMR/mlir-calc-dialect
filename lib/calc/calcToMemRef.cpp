#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CALCTOMEMREFPASS
#include "calc/calcPasses.h.inc"

namespace {
class convertStackOp : public mlir::OpRewritePattern<calc::stackOp> {
    using mlir::OpRewritePattern<calc::stackOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(calc::stackOp op, mlir::PatternRewriter &rewriter) const override {

        // get the location
        mlir::Location loc = op.getLoc();

        // get all the operands from the stack op
        std::vector<mlir::Value> operands(op.getOperands().begin(), op.getOperands().end());
        
        // pop the last buffer as it is for result
        operands.pop_back();

        // get the result buffer
        mlir::Value result = op.getOperands().back();

        // Get the input/result types.
        mlir::MemRefType inputType = mlir::cast<mlir::MemRefType>(operands[0].getType());
        mlir::MemRefType resultType = mlir::cast<mlir::MemRefType>(result.getType());
    
        // Initiaize a dim value to 0 and if the dim attribute is present, update the dim value accordingly.
        int64_t dimVal = 0;
        mlir::Attribute dimAttr = op.getDimAttr();
        if (dimAttr) {
            dimVal = llvm::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
        }

    
        // normalize negative dim first
        if (dimVal < 0) dimVal += inputType.getRank() + 1;
    
        // defensive bounds check
        if (dimVal < 0 || dimVal > inputType.getRank())
            return rewriter.notifyMatchFailure(op, "dim out of range");

        // compute the offsets, sizes, strides to use subview of the result buffer to insert the input 
        int64_t offset = 0;
        for (mlir::Value input : operands) {
            llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;

            for (int64_t i = 0; i < resultType.getRank(); i++) {
                if (dimVal == i) {
                    offsets.push_back(rewriter.getIndexAttr(offset));
                    sizes.push_back(rewriter.getIndexAttr(1));
                    strides.push_back(rewriter.getIndexAttr(1));
                } else {
                    int64_t inputDim = i < dimVal ? i : i - 1;
                    offsets.push_back(rewriter.getIndexAttr(0));
                    sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[inputDim]));
                    strides.push_back(rewriter.getIndexAttr(1));
                }
            }

            // get the memref result type with shape, strided, and offset 
            // result shape = 2x4x4 input1 shape is 4x4
            // (eg. memref<1x4x4xi32, strided<[16, 4, 1], offset: 0>>)
            mlir::MemRefType subviewType = mlir::memref::SubViewOp::inferResultType(resultType, offsets, sizes, strides);

            // create the subview of the result tensor to insert a copy of input
            mlir::Value subview = mlir::memref::SubViewOp::create(rewriter, loc, subviewType, result, offsets, sizes, strides);

            // create a reassociation vector for expandshape op 
            // eg. input1 shape is 4x4 -> 1x4x4
            // reassoc = [[0,1][2]]
            mlir::SmallVector<mlir::ReassociationIndices> reassoc;
            int64_t outIdx = 0;
            for (int64_t i = 0; i < inputType.getRank(); i++) {
                if (i == dimVal) {
                    reassoc.push_back({outIdx, outIdx + 1});
                    outIdx += 2;
                } else {
                    reassoc.push_back({outIdx});
                    outIdx += 1;
                }
            }

            // handle dimVal == inputRank (append at end)
            if (dimVal == inputType.getRank()) {
                reassoc.back().push_back(outIdx);
            }

            // get the input shape
            std::vector<int64_t> expandedShape = inputType.getShape().vec();
            
            // pad 1 to increase the rank
            expandedShape.insert(expandedShape.begin() + dimVal, 1);
            
            // create the memref type for the new shape
            mlir::MemRefType expandType = mlir::MemRefType::get(expandedShape, inputType.getElementType());

            // expand the input type so it would match the subview of the result
            mlir::Value expanded = mlir::memref::ExpandShapeOp::create(rewriter, loc, expandType, input, reassoc);

            // copy the input to result's subview
            mlir::memref::CopyOp::create(rewriter, loc, expanded, subview);

            offset += 1;
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};
}

namespace calc{
class CalcToMemRefPass : public impl::CalcToMemRefPassBase<CalcToMemRefPass> {
public:
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect, calc::calcDialect>();
    }
    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        mlir::RewritePatternSet patterns(&getContext());
        
        // Adding legal dialect
        target.addLegalDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                               mlir::bufferization::BufferizationDialect, calc::calcDialect>();
        // Adding calc.stack Op as Illegal Op
        target.addIllegalOp<stackOp>();

        // Applying the conversion pattern for calc.stack Op
        patterns.add<convertStackOp>(&getContext());
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))){
            signalPassFailure();
        }
    }
};

std::unique_ptr<mlir::Pass> createCalcToMemRefPass() {
    return std::make_unique<CalcToMemRefPass>();
}
}

