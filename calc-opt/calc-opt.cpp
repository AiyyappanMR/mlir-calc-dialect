#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h" 
#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h" 

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    calc::registerCalcPasses();
    mlir::DialectRegistry registry;
    registry.insert<calc::calcDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::tosa::TosaDialect,mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::affine::AffineDialect
                    >();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Calc optimizer\n", registry));
}


//-------------------------------------------------------
// Another way of defining opt still in progress
//-------------------------------------------------------
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/Parser/Parser.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/IR/OwningOpRef.h"   // for owningopref
// #include "mlir/IR/BuiltinOps.h"    // for ModuleOp


// #include "llvm/Support/CommandLine.h"  // for cmd line input
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/raw_ostream.h" 

// #include "calc/calcDialect.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"

// namespace cl = llvm::cl;
// static cl::opt<std::string> inputFilename(cl::Positional,
//                                         cl::desc("<input hello file>"),
//                                         cl::init("-"),
//                                         cl::value_desc("filename"));

// int loadMLIR(mlir::MLIRContext &context,
//              mlir::OwningOpRef<mlir::ModuleOp> &module) {
//   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
//       llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
//   if (std::error_code ec = fileOrErr.getError()) {
//     llvm::errs() << "Could not open input file: " << ec.message() << "\n";
//     return -1;
//   }

//   llvm::SourceMgr sourceMgr;
//   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
//   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
//   if (!module) {
//     llvm::errs() << "Error can't load file " << inputFilename << "\n";
//     return 3;
//   }
//   return 0;
// }

// int main(int argc, char **argv){
//     mlir::registerMLIRContextCLOptions();

//     cl::ParseCommandLineOptions(argc, argv, "calc compiler\n");
//     mlir::MLIRContext context;
//     context.getOrLoadDialect<calc::calcDialect>();
//     context.getOrLoadDialect<mlir::func::FuncDialect>();
//     context.getOrLoadDialect<mlir::arith::ArithDialect>();

//     mlir::OwningOpRef<mlir::ModuleOp> module;
//     if (int error = loadMLIR(context, module)) {
//         return error;
//   }

//     return 0;
// }
