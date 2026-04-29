# This file contains pytest test cases to test the python bindings for the calc dialect by building a simple IR with the 
# calc.addt op, lowering it to llvm IR, executing it with the execution engine, and comparing the results against 
# PyTorch's add operator.    

import sys
import gc
import sys
import ctypes
import subprocess
import pytest
import torch
import numpy as np

# calc_mlir is a self-contained package that bundles MLIR core + our dialect.
CALC_PYTHON_PATH = "/home/mcw/Desktop/mlir-calc-dialect/build/python_packages/calc_mlir"
if CALC_PYTHON_PATH not in sys.path:
    sys.path.insert(0, CALC_PYTHON_PATH)

# Importing the necessary components from calc_mlir to build, lower, and execute the IR. 
import calc_mlir 
from calc_mlir.ir import Context, Module
from calc_mlir.passmanager import PassManager
from calc_mlir.execution_engine import ExecutionEngine
from calc_mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
from calc_mlir._mlir_libs._calcMlir import register_dialect   

#---------------------------------------------------------------------------------
#                 Calc.addt execution
#---------------------------------------------------------------------------------

# pass pipeline to lower tosa to llvm IR
def CalcToLLVM(module):
    pm = PassManager.parse(
        "builtin.module("
            "calc-to-tosa,"  # custom pass to lower from calc dialect to tosa dialect
            "func.func(tosa-to-linalg),"
            "func.func(tosa-to-arith),"
            "func.func(tosa-to-tensor),"
            "convert-tensor-to-linalg,"
            "one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs-from-loops=true},"
            "func.func(convert-linalg-to-loops),"
            "func.func(convert-scf-to-cf),"
            "expand-strided-metadata,"
            "convert-index-to-llvm,"              
            "finalize-memref-to-llvm,"
            "convert-math-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-cf-to-llvm,"                 
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


# calc.addt IR template which takes the tensor type and shape from the argument
def build_ir(tensor_type):
    tensor_type = f"tensor<{tensor_type}>"
    
    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_addt(%arg0: {tensor_type}, %arg1: {tensor_type}) -> {tensor_type} attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.addt"(%arg0, %arg1) : ({tensor_type}, {tensor_type}) -> {tensor_type}
  return %0 : {tensor_type}
}}
"""
    return ir

# Main function which calls all the other function to lower the calc.addt op to llvm IR and executes them.
def run_addt(input1, input2, tensor_type):
    
    # lowers calc to tosa
    calc_ir = build_ir(tensor_type)

    # Initializing a result array to store the output
    result = np.zeros_like(input1.numpy())
    with Context() as ctx:

        register_dialect(ctx)

        # parse the lowered IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(CalcToLLVM(module))

        # Converts the torch -> numpy -> memref
        mem_input1 = get_ranked_memref_descriptor(input1.numpy())
        mem_input2 = get_ranked_memref_descriptor(input2.numpy())
        mem_result = get_ranked_memref_descriptor(result)

        # pointer to the memref struct  instead of value
        final_input1 = ctypes.pointer(ctypes.pointer(mem_input1))
        final_input2 = ctypes.pointer(ctypes.pointer(mem_input2))
        final_result = ctypes.pointer(ctypes.pointer(mem_result))

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_addt", final_result, final_input1, final_input2)
        
        # convert the result memref back to numpy
        np_result = ranked_memref_to_numpy(ctypes.pointer(mem_result))

        return np_result
    

#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases which has the following parameter's input1, input2, and tensortype
ADDT_TEST_CASES = [
    pytest.param(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype = torch.float64),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype = torch.float64),
        "2x2xf64",
        id = "2x2xf64_basic",
    ),
    pytest.param(
        torch.tensor([10.0, 0.0], dtype = torch.float64),
        torch.tensor([6.0, 6.0], dtype = torch.float64),
        "2xf64",
        id = "2xf64_negatives_and_mixed",
    ),
    pytest.param(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype = torch.float64),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype = torch.float64),
        "2x2xf64",
        id = "2x2xf64_zeros",
    ),
    pytest.param(
        torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype = torch.float64),
        torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype = torch.float64),
        "2x3xf64",
        id = "2x3xf64_different_shape",
    ),
    pytest.param(
        torch.tensor([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]],[[5,5,5],[6,6,6]]], dtype = torch.float64),
        torch.tensor([[[10,10,10],[20,20,20]],[[30,30,30],[40,40,40]],[[50,50,50],[60,60,60]]], dtype = torch.float64),
        "3x2x3xf64",
        id = "3x2x3xf64_different_shape",
    )
]

@pytest.mark.parametrize("input1, input2, tensor_type", ADDT_TEST_CASES)
def test_addt(input1, input2, tensor_type):
    
    # actual and calc outputs 
    expected = torch.add(input1, input2).numpy()
    actual = run_addt(input1, input2, tensor_type)

    print(f"\n[{tensor_type}] actual  : {actual}")
    print(f"[{tensor_type}] expected: {expected}")

    # verify lowering didnt fail
    assert actual is not None, "run_addt returned None — lowering failed"
    
    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
