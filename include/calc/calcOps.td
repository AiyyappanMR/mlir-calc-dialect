#ifndef CALC_OPS
#define CALC_OPS

//include "mlir/IR/OpBase.td"
//include "mlir/Interfaces/SideEffectInterfaces.td"
include "calcDialect.td"

def calc_constOp : calc_Op<"const", [Pure]> {
    let summary = "To get literal value";
    let description = [{
        Constant operation turns a literal integer value into an SSA value.
    }];
    let arguments = (ins AnyIntegerAttr:$value);
    let results = (outs AnyIntegerType:$result);
    let builder = [
        OpBuilder<(ins "mlir::IntegerAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];
    let assemblyFormat = "$value attr-dict";
}

def calc_addOp : calc_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "Add two values";
    let description = [{
        Takes two integer operands and returns their sum.
    }];
    let arguments = (ins AnyTypeOf<[I32, I64]>:$input1, AnyTypeOf<[I32, I64]>:$input2);
    let results = (outs AnyTypeOf<[I32, I64]>:$result);
    let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($result)";
}

def calc_mulOp : calc_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "Multiply two values";
    let description = [{
        Takes two integer operands and returns their product.
    }];
    let arguments = (ins AnyTypeOf<[I32, I64]>:$input1, AnyTypeOf<[I32, I64]>:$input2);
    let results = (outs AnyTypeOf<[I32, I64]>:$result);
    let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($result)";
}

def calc_printOp : calc_Op<"print", []> {
    let summary = "Prints the result";
    let description = [{Takes one integer operand and prints it.}];
    let arguments = (ins AnyIntegerType:$input1);
    let assemblyFormat = "$input1 `:` type($input1) attr-dict ";
}

#endif