#ifndef CALC_PATTERNS_TD
#define CALC_PATTERNS_TD

include "calc/calcOps.td"
include "mlir/IR/PatternBase.td"

def HasOneUse : Constraint<CPred<"$_self.hasOneUse()">,"Has one use">;

def FuseAddMul : Pat<
    (calc_addtOp $input, (calc_multOp:$mul $tensor1, $tensor2)),
    (calc_addcmulOp $input, $tensor1, $tensor2, (NativeCodeCall<"mlir::Attribute{}">)),
    [(HasOneUse:$mul)]
>;

#endif // CALC_PATTERNS_TD