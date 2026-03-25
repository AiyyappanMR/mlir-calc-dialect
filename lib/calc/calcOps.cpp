#include "mlir/IR/OpImplementation.h"
#include "calc/calcDialect.h"
#include "calc/calcOps.h"

using namespace mlir;
using namespace calc;

#define GET_OP_CLASSES
#include "calc/calcOps.cpp.inc" 