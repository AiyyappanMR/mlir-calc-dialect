#ifndef CALC_C_DIALECTS_H
#define CALC_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare the C API for registering the Calc dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Calc, calc);

#ifdef __cplusplus
}
#endif

#endif // CALC_C_DIALECTS_H