add_llvm_executable(calc-opt calc-opt.cpp)

llvm_update_compile_flags(calc-opt)

target_link_libraries(calc-opt
    PRIVATE
    MLIRCalc
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRRegisterAllPasses
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTosaDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRAffineDialect
)

mlir_check_all_link_libraries(calc-opt)