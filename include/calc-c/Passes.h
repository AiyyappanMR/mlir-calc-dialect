#ifndef CALC_C_PASSES_H
#define CALC_C_PASSES_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Include the generated C-API header for the passes, which defines the registration function
#include "calc/calcPasses.capi.h.inc"

#ifdef __cplusplus
}
#endif

#endif // CALC_C_PASSES_H
