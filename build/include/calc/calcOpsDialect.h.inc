/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: calcOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace calc {

/// A simple custom dialect for arithmetic operations
/// This dialect defines basic arithmetic operations 
///         such as addition and multiplication.
class calcDialect : public ::mlir::Dialect {
  explicit calcDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~calcDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("calc");
  }
};

} // namespace calc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::calc::calcDialect)
