#ifndef CALC_C_REGISTRATION_H
#define CALC_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare the C API for registering calc dialect and all dependent dialects.
MLIR_CAPI_EXPORTED void calcMlirRegisterAllDialects(MlirContext context);

// Just the declaration for the Python bindings to use
MLIR_CAPI_EXPORTED void calcRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif