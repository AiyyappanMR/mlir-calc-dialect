add_mlir_dialect_library(MLIRCalc
    calcDialect.cpp
    calcOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/calc

    DEPENDS
    MLIRcalcOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)