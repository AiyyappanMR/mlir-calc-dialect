add_mlir_dialect_library(MLIRCalc
    broadcast.cpp
    calcDialect.cpp
    calcOps.cpp
    calcToArith.cpp
    calcToTosa.cpp
    calcBufferize.cpp
    calcToMemRef.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/calc

    DEPENDS
    MLIRcalcOpsIncGen
    MLIRCalcTransformsIncGen
    MLIRCalcPatternsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRTosaDialect
    MLIRTransforms
    MLIRPass
)