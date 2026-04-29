#include "calc-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "calc/calcDialect.h"

// Register the dialects.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Calc, calc, ::calc::calcDialect)