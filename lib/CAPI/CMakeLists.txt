add_mlir_public_c_api_library(MLIRCalcCAPI
  Dialects.cpp
  Registration.cpp
  Transforms.cpp

  # This tells the macro where to find your .h files 
  # for the public interface
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/calc-c/

  # Ensures symbols from linked libs are bundled into this C-API
  ENABLE_AGGREGATION

  # Ensure tablegen headers are ready before compilation
  DEPENDS
  MLIRCalcTransformsIncGen
  MLIRcalcOpsIncGen

  LINK_LIBS PUBLIC
  # Core MLIR C-API dependencies
  MLIRIR
  MLIRSupport
  MLIRCAPIIR

  # Your custom C++ library
  MLIRCalc
)