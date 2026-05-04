#ifndef CALCDIALECT_H
#define CALCDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

#include "calc/calcOpsDialect.h.inc"

namespace calc {
namespace OpTrait {
template <typename ConcreteType>
class OperandsMustbeFloat : public mlir::OpTrait::TraitBase<ConcreteType, OperandsMustbeFloat> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
            // Loop through all operands 
            for (mlir::Value operand : op->getOperands()) {
                // This 'unwraps' tensor<f32> to f32
                mlir::Type elemType = mlir::getElementTypeOrSelf(operand);
                
                if (!mlir::isa<mlir::FloatType>(elemType)) {
                    return op->emitOpError("requires all operands to have floating-point element types");
                }
            }
            return mlir::success();
        }
};

}
}

#endif