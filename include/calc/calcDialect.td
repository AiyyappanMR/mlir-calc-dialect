#ifndef CALC_DIALECT
#define CALC_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def calc_Dialect : Dialect {
    let name = "calc";
    let summary = "A simple custom dialect for arithmetic operations";
    let description = [{ 
        This dialect defines basic arithmetic operations 
        such as addition and multiplication.
    }];
    let cppNamespace = "::calc";
}

class calc_Op<string mnemonic, list<Trait> traits = []> 
    : Op<calc_Dialect, mnemonic, traits>;

#endif
