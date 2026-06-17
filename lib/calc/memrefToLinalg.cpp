#include "calc/calcPasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#define GEN_PASS_DEF_MEMREFTOLINALGPASS
#include "calc/calcPasses.h.inc"

namespace {
class convertCopyOp : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
    using mlir::OpRewritePattern<mlir::memref::CopyOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::memref::CopyOp op, mlir::PatternRewriter &rewriter) const override {

        // get the location
        mlir::Location loc = op.getLoc();

        // get the input types
        mlir::Value source = op.getSource();
        mlir::MemRefType SourceTy = mlir::cast<mlir::MemRefType>(source.getType());
        mlir::Value target = op.getTarget();
        mlir::MemRefType TargetTy = mlir::cast<mlir::MemRefType>(target.getType());

        // Verify that the source and target memref has matching ranks
        if (SourceTy.getRank() != TargetTy.getRank()){
            return rewriter.notifyMatchFailure(op, "Source and Target rank doesnt match");
        }

        // Create Affine Maps for source and target
        mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(SourceTy.getRank(), rewriter.getContext());
        llvm::SmallVector<mlir::AffineMap> indexingMaps = {identityMap, identityMap};

        // Create the IteratorType
        llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(SourceTy.getRank(), mlir::utils::IteratorType::parallel);

        // creates linalg generic Op
        auto linalg = mlir::linalg::GenericOp::create(rewriter, loc, mlir::TypeRange{}, source, target, indexingMaps, iteratorTypes,
            [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
                mlir::linalg::YieldOp::create(b, loc, args[0]);
            });

        // erase the existing memref.copy Op
        rewriter.eraseOp(op);

        return mlir::success();
    }
};
}

namespace calc{
class MemRefToLinalgPass : public impl::MemRefToLinalgPassBase<MemRefToLinalgPass> {
public:
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect>();
    }
    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        mlir::RewritePatternSet patterns(&getContext());
        
        // Adding legal dialect
        target.addLegalDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect>();
        // Adding memref.Copy Op as Illegal Op
        target.addIllegalOp<mlir::memref::CopyOp>();

        // Applying the conversion pattern for memref.Copy Op
        patterns.add<convertCopyOp>(&getContext());
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))){
            signalPassFailure();
        }
    }
};

std::unique_ptr<mlir::Pass> createMemRefToLinalgPass() {
    return std::make_unique<MemRefToLinalgPass>();
}
}
