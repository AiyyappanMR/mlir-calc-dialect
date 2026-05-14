# This file contains pytest tests for functional testing of minimum op with respect to pytorch's minimum operator. 

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
#                 Calc.minimum execution
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


# calc.minimum IR template which takes the tensor type and shape from the argument
def build_ir(tensor_type):
    tensor_type = f"tensor<{tensor_type}>"
    
    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_minimum(%arg0: {tensor_type}, %arg1: {tensor_type}) -> {tensor_type} attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.minimum"(%arg0, %arg1) : ({tensor_type}, {tensor_type}) -> {tensor_type}
  return %0 : {tensor_type}
}}
"""
    return ir

# Main function which calls all the other function to lower the calc.minimum op to llvm IR and executes them.
def run_minimum(input1, input2, tensor_type):
    
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
        execution_engine.invoke("test_minimum", final_result, final_input1, final_input2)
        
        # convert the result memref back to numpy
        np_result = ranked_memref_to_numpy(ctypes.pointer(mem_result))

        return np_result
    

#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases which has the following parameter's input1, input2, and tensortype
MINIMUM_TEST_CASES = [
    # 1D f32 — fixed values including negatives
    (
        torch.tensor([1.0, 3.0, -2.0, 5.0]),
        torch.tensor([4.0, -1.0, 3.0, 2.0]),
        "4xf32",
    ),
    # 2D f32 — random 3x3
    (
        torch.randn(3, 3, dtype=torch.float32),
        torch.randn(3, 3, dtype=torch.float32),
        "3x3xf32",
    ),
    # 1D f32 — random 6 elements
    (
        torch.randn(6, dtype=torch.float32),
        torch.randn(6, dtype=torch.float32),
        "6xf32",
    ),
    # 1D f64 — random 8 elements
    (
        torch.randn(8, dtype=torch.float64),
        torch.randn(8, dtype=torch.float64),
        "8xf64",
    ),
    # 2D i64 — random 2x4
    (
        torch.randint(0, 10, (2, 4), dtype=torch.int64),
        torch.randint(0, 10, (2, 4), dtype=torch.int64),
        "2x4xi64",
    ),
    # 3D i32 — random 2x2x2
    (
        torch.randint(0, 10, (2, 2, 2), dtype=torch.int32),
        torch.randint(0, 10, (2, 2, 2), dtype=torch.int32),
        "2x2x2xi32",
    ),
]

@pytest.mark.parametrize("input1, input2, tensor_type", MINIMUM_TEST_CASES)
def test_minimum(input1, input2, tensor_type):
    
    # print the input tensors 
    print(f"\n input1: {input1}")
    print(f" input2: {input2}")

    # actual and calc outputs 
    expected = torch.minimum(input1, input2).numpy()
    actual = run_minimum(input1, input2, tensor_type)

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)
    # verify lowering didnt fail
    assert actual is not None, "run_minimum returned None — lowering failed"
    
    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)