add_mlir_library(MLIRCalcCAPI
  Passes.cpp

  DEPENDS
  MLIRCalcPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCalc
  MLIRCAPIIR
)
