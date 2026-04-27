add_mlir_dialect_library(MLIRCalc
    calcDialect.cpp
    calcOps.cpp
    calcToArith.cpp
    calcToTosa.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/calc

    DEPENDS
    MLIRcalcOpsIncGen
    MLIRCalcPassesIncGen
    CalcPythonOpsGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTosaDialect
    MLIRTransforms
    MLIRPass
)