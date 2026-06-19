#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CALCBUFFERIZEPASS
#include "calc/calcPasses.h.inc"

namespace {
class StackBufferizePattern : public mlir::OpConversionPattern<calc::stackOp> {
    using mlir::OpConversionPattern<calc::stackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(calc::stackOp op, calc::stackOpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();

        // getting the result to extract its type
        mlir::RankedTensorType resultType = mlir::cast<mlir::RankedTensorType>(op.getResult()[0].getType());

        // creating a mmeref type
        mlir::MemRefType memrefType = mlir::MemRefType::get(resultType.getShape(), resultType.getElementType());
        
        // allocating new empty buffer to store the result
        mlir::Value alloc = mlir::memref::AllocOp::create(rewriter, loc, memrefType);

        // create a vector of all the inputs and the new allocated buffer
        mlir::SmallVector<mlir::Value> newOperands(adaptor.getOperands().begin(), adaptor.getOperands().end());
        newOperands.push_back(alloc);

        // create a calc.stack op with zero reults and pass the result buffer as operand to it.
        calc::stackOp::create(rewriter, op.getLoc(), mlir::TypeRange{}, newOperands, op->getAttrs());
        
        rewriter.replaceOp(op, alloc);
        return mlir::success();
    }
};
}

namespace calc{
class CalcBufferizePass : public impl::CalcBufferizePassBase<CalcBufferizePass> {
public:
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect, calc::calcDialect>();
    }
    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        mlir::TypeConverter typeConverter;

        // Bufferize all tensor types to memref types, and keep non-tensor types unchanged.
        typeConverter.addConversion([](mlir::Type type) { return type; });
        typeConverter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
            return mlir::MemRefType::get(type.getShape(), type.getElementType());
        });

        // Add source and target materialization for the type conversion.
        typeConverter.addSourceMaterialization([](mlir::OpBuilder &builder, mlir::TensorType type, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
            return mlir::bufferization::ToTensorOp::create(builder, loc, type, inputs[0], /*restrict=*/true, /*writable=*/false);
        });
        typeConverter.addTargetMaterialization([](mlir::OpBuilder &builder, mlir::BaseMemRefType type, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
            return mlir::bufferization::ToBufferOp::create(builder, loc, type, inputs[0]);
        });

        // Mark operations as legal or illegal.
        target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect, calc::calcDialect>();
        target.addDynamicallyLegalDialect<calc::calcDialect>([&](mlir::Operation *op) {
            return typeConverter.isLegal(op);
        });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<StackBufferizePattern>(typeConverter, &getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))){
            return signalPassFailure();
        }
        
    }
};

std::unique_ptr<mlir::Pass> createCalcBufferizePass() {
  return std::make_unique<CalcBufferizePass>();
}


} // namespace calc