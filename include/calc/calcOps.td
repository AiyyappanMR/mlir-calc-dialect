#ifndef CALC_OPS
#define CALC_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/IR/OpBase.td"
//include "mlir/Bytecode/BytecodeOpInterface.td"
include "calc/calcDialect.td"

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

def calc_addtOp : calc_Op<"addt", [Pure, Commutative, SameOperandsAndResultRank, SameOperandsAndResultElementType, ResultsBroadcastableShape]> {
    let summary = "Add two tensor values";
    let description = [{
        Adds two tensors of identical shape and element type.
    }];
    let arguments = (ins RankedTensorOf<[AnyInteger, AnyFloat]>:$input1,RankedTensorOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
    let hasCanonicalizer = 1;
}

def calc_multOp : calc_Op<"mult", [Pure, Commutative, SameOperandsAndResultRank, SameOperandsAndResultElementType, ResultsBroadcastableShape]> {
    let summary = "Multiply two tensor values";
    let description = [{
        Multiplies two tensors of identical shape and element type.
    }];
    let arguments = (ins RankedTensorOf<[AnyInteger, AnyFloat]>:$input1,RankedTensorOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def calc_addcmulOp : calc_Op<"addcmul",[Pure, SameOperandsAndResultElementType, SameOperandsAndResultRank, ResultsBroadcastableShape]> {
    let summary = "Add scaled element-wise multiplication of two tensors to input";
    let description = [{Performs the element-wise multiplication of tensor1 by tensor2, 
                        multiplies the result by the scalar value and adds it to input.
                        '''
                        Output = input + value * (tensor1 * tensor2)
                        '''
                        }];
    let arguments = (ins RankedTensorOf<[AnyInteger, AnyFloat]>:$input, RankedTensorOf<[AnyInteger, AnyFloat]>:$tensor1, 
                        RankedTensorOf<[AnyInteger, AnyFloat]>:$tensor2, OptionalAttr<AnyAttrOf<[F32Attr, F64Attr, I32Attr, I64Attr]>>:$value);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
    let builders = [  
        OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$input, "mlir::Value":$tensor1, "mlir::Value":$tensor2), 
        [{build($_builder, $_state, resultType, input, tensor1, tensor2, mlir::Attribute{});}]>
    ];
}

def calc_logaddexp2Op : calc_Op<"logaddexp2", [Pure, Commutative, SameOperandsAndResultElementType, SameOperandsAndResultRank, ResultsBroadcastableShape]> {
    let summary = "Add two tensors in log space with base 2";
    let description = [{
    Computes the logarithm of the sum of exponentiations of the inputs in log space with base 2.
    '''
    Output = log2(2^input1 + 2^input2)
    '''
    }];
    let arguments = (ins RankedTensorOf<[AnyFloat]>:$input1, RankedTensorOf<[AnyFloat]>:$input2);
    let results = (outs RankedTensorOf<[AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def calc_minimumOp : calc_Op<"minimum", [Pure, Commutative, SameOperandsAndResultElementType, ResultsBroadcastableShape]> {
    let summary = "performs element-wise minimum of two tensors";
    let description = [{
    Computes the element-wise minimum of two tensors of identical shape and element type.
    '''
    Output = calc.minimum([1,2,3], [2,1,4]) = [1,1,3]
    '''
    }];
    let arguments = (ins RankedTensorOf<[AnyInteger, AnyFloat]>:$input1, RankedTensorOf<[AnyInteger, AnyFloat]>:$input2);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def calc_entrOp : calc_Op<"entr", [Pure]> {
    let summary = "Computes the element-wise entropy of a tensor";
    let description = [{
    Computes the element-wise entropy of a tensor, defined as -x * log(x) for x > 0, 0 for x == 0, and -inf for x < 0.
    '''
    Output = calc.entr([0.5, -1.0, 0]) = [0.34657359, -inf, 0.0]
    '''
    }];
    let arguments = (ins RankedTensorOf<[F32, F64, I32, I64]>:$input);
    let results = (outs RankedTensorOf<[F32, F64]>:$result);
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def calc_splitOp : calc_Op<"split", [Pure, SameOperandsAndResultElementType]> {
    let summary = "Splits a tensor into chunks of specified sizes along a dimension";
    let description = [{
        Splits the input tensor into `len(split_sizes)` chunks along `dim`,
        where each chunk has the size specified by the corresponding entry in
        `split_sizes`. The sizes must sum exactly to the size of the input
        tensor along `dim`.

        `dim` defaults to 0 if not specified. Negative `dim` indexing is
        supported and is normalized as `dim + rank` before use.

        Example:
          %a, %b = calc.split %input {split_sizes = array<i64: 1, 4>, dim = 0 : si32}
                   : (tensor<5x4xf32>) -> (tensor<1x4xf32>, tensor<4x4xf32>)
    }];
    let arguments = (ins
        RankedTensorOf<[I32, I64, F32, F64]>:$input,
        DenseI64ArrayAttr:$split_sizes,
        OptionalAttr<SI32Attr>:$dim
    );
    let results = (outs Variadic<RankedTensorOf<[I32, I64, F32, F64]>>:$results);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` functional-type($input, $results)";
}

def calc_catmuladdOp : calc_Op<"catmuladd", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultRank, AttrSizedOperandSegments]>{
    let summary = "Concatenates two variadic tensor groups separately, multiplies the results element-wise, and optionally adds a scale tensor to the product";
    let description = [{
    Concatenates the `inputs1` tensors along `dim` to produce `cat_a`, and
    concatenates the `inputs2` tensors along `dim` to produce `cat_b`.
    The two results are multiplied element-wise to produce `mul_out`.
    If `scale` is provided, it is added element-wise to `mul_out` to produce
    the final result. If `scale` is absent, `mul_out` is returned directly.

    The two variadic groups may have different numbers of operands, but the
    concatenated shapes of `inputs1` and `inputs2` must be identical since the
    results are multiplied element-wise.
    }];
    let arguments = (ins Variadic<RankedTensorOf<[AnyInteger, AnyFloat]>>:$inputs1, Variadic<RankedTensorOf<[AnyInteger, AnyFloat]>>:$inputs2, Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$scale, OptionalAttr<SI32Attr>:$dim);
    let results = (outs RankedTensorOf<[AnyInteger, AnyFloat]>:$result);
    let hasVerifier = 1;
    let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
    }];
}

def calc_prodOp : calc_Op<"prod",[Pure, SameOperandsAndResultElementType]>{
    let summary = "Computes the product of all elements in the input tensor";
    let description = [{
    Computes the product of elements in the input tensor.

    If `dim` (i32) is specified, the product is computed along that axis.
    If `keepdim` (bool, default: false) is true, the reduced dimension is retained as size 1.
    If `keepdim` is false, the reduced dimension is removed and rank is reduced by 1.
    `keepdim` is only valid when `dim` is specified.

    If `dim` is not specified, product of all elements is returned as a scalar.
    Negative `dim` indexing is supported. Empty tensor returns 1.
    }];
    let arguments = (ins RankedTensorOf<[I32, I64, F32, F64]>:$input, OptionalAttr<SI32Attr>:$dim, OptionalAttr<BoolAttr>:$keepdim);
    let results = (outs RankedTensorOf<[I32, I64, F32, F64]>:$result);
    let hasVerifier = 1;
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";   
}

def calc_softmaxOp : calc_Op<"softmax", [Pure, SameOperandsAndResultType]> {
    let summary = "Computes the softmax of the input tensor along the specified dimension";
    let description = [{
    Computes the softmax of the input tensor along the specified dimension.

    The softmax function is defined as:
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

    If `dim` (i32) is specified, softmax is computed along that axis.
    Negative `dim` indexing is supported. Dim must be present and valid. 
    }];
    let arguments = (ins RankedTensorOf<[F32, F64]>:$input, SI32Attr:$dim);
    let results = (outs RankedTensorOf<[F32, F64]>:$result);
    let hasVerifier = 1;
    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";  
}

def calc_stackOp : calc_Op<"stack", [Pure]> {
    let summary = "Stacks a list of tensors along a new dimension";
    let description = [{
    Stacks a list of tensors along a new dimension specified by `dim` (i32).

    All input tensors must have the same shape and element type.
    The output tensor will have one more dimension than the input tensors, with size equal to the number of input tensors along the `dim` axis.

    Negative `dim` indexing is supported. Dim must be present and valid.
    }];
    let arguments = (ins Variadic<AnyShaped>:$inputs, OptionalAttr<SI32Attr>:$dim);
    let results = (outs Variadic<AnyShaped>:$result);
    let hasVerifier = 1;
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

def calc_testOp : calc_Op<"test", [Pure, OperandsMustbeFloat]>{
    let summary = "Testing operation for the OperandsMustbeFloat trait verification";
    let description = [{
        An internal testing operation designed explicitly to validate the behavior
        of the `OperandsMustbeFloat` trait constraint.
    }];
    let arguments = (ins 
        AnyTypeOf<[AnyInteger, AnyFloat]>:$input1, 
        AnyTypeOf<[AnyInteger, AnyFloat]>:$input2
    );
    let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
    let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($input1) `,` type($input2) `->` type($result)";
}

#endif
