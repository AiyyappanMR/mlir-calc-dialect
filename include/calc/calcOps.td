#ifndef CALC_OPS
#define CALC_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  
//include "mlir/Bytecode/BytecodeOpInterface.td"
include "calcDialect.td"

def calc_constOp : calc_Op<"const", [Pure]> {
    let summary = "To get literal value";
    let description = [{
        Constant operation turns a literal integer value into an SSA value.
    }];
    let arguments = (ins AnyAttrOf<[I32Attr, I64Attr]>:$value);
    let results = (outs AnyTypeOf<[I32, I64]>:$result);
    let builders = [
        OpBuilder<(ins "mlir::IntegerAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];
    //let assemblyFormat = "attr-dict $value";
}

def calc_addOp : calc_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "Add two scaler values";
    let description = [{
        Adds two scalar integer or float values of the same type.
    }];
    let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$input1, AnyTypeOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($result)";
}

def calc_addtOp : calc_Op<"addt", [Pure, Commutative, SameOperandsAndResultRank, SameOperandsAndResultElementType]> {
    let summary = "Add two tensor values";
    let description = [{
        Adds two tensors of identical shape and element type.
    }];
    let arguments = (ins RankedTensorOf<[AnyInteger, AnyFloat]>:$input1,RankedTensorOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def calc_mulOp : calc_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "Multiply two values";
    let description = [{
        Takes two integer operands and returns their product.
    }];
    let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$input1, AnyTypeOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($result)";
}

def calc_printOp : calc_Op<"print", []> {
    let summary = "Prints the result";
    let description = [{Takes one integer operand and prints it.}];
    let arguments = (ins AnyTypeOf<[I32, I64]>:$input1);
    let assemblyFormat = "$input1 `:` type($input1) attr-dict ";
}

#endif
