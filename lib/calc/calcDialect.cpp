#include "calc/calcDialect.h"
#include "calc/calcOps.h"

using namespace mlir;
using namespace calc;

void calcDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "calc/calcOps.cpp.inc"
    >();
}

//#include "calc/calcOpsDialect.h.inc"