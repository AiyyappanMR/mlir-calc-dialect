#ifndef CALC_PASSES_H
#define CALC_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"
//namespace mlir{
namespace calc{

#define GEN_PASS_DECL
#include "calc/calcPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "calc/calcPasses.h.inc"

}
//}

#endif
