add_mlir_dialect_library(MLIRCalc
    broadcast.cpp
    calcDialect.cpp
    calcOps.cpp
    calcToArith.cpp
    calcToTosa.cpp
    memrefToLinalg.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/calc

    DEPENDS
    MLIRcalcOpsIncGen
    MLIRCalcTransformsIncGen
    MLIRCalcPatternsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRTosaDialect
    MLIRTransforms
    MLIRPass
)