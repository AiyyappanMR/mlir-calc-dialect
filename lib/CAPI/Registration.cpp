#include "calc-c/Registration.h"

#include "calc/calcDialect.h"   
#include "calc/calcPasses.h"        
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"

// Standard MLIR Dialect includes
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

void calcMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  
  // Register your custom dialect and all dependencies
  registry.insert<
      ::calc::calcDialect,
      mlir::func::FuncDialect,
      mlir::arith::ArithDialect,
      mlir::tosa::TosaDialect,
      mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect
  >();

  // Append the registry to the context
  unwrap(context)->appendDialectRegistry(registry);
  
  // load them to ensure they are available for parsing
  unwrap(context)->loadAllAvailableDialects();
}

void calcRegisterAllPasses() {
  // Call your internal C++ pass registration logic
  ::calc::registerCalcPasses();
}