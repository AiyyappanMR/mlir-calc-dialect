add_mlir_dialect_library(MLIRCalc
    calcDialect.cpp
    calcOps.cpp
    calcToArith.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/calc

    DEPENDS
    MLIRcalcOpsIncGen
    MLIRCalcPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)