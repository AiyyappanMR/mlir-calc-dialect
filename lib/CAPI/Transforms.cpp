#include "mlir/CAPI/Pass.h"
#include "calc/calcPasses.h"

// Must include the declarations as they carry important visibility attributes.
#include "calc/calcTransforms.capi.h.inc"

using namespace calc;

#ifdef __cplusplus
extern "C" {
#endif

// Include the generated C API function definitions.
#include "calc/calcTransforms.capi.cpp.inc"

#ifdef __cplusplus
}
#endif