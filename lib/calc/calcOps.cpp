#include "calc/calcOps.h"
#include "calc/calcDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace calc;

#define GET_OP_CLASSES
#include "calc/calcOps.cpp.inc"