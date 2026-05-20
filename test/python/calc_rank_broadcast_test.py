# This file contains pytest tests for functional testing of minimum op with respect to pytorch's minimum operator with operands of differenr ranks. 

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
            "calc-broadcast,"       
            "calc-to-tosa,"
            "func.func(tosa-to-linalg),"
            "func.func(tosa-to-arith),"
            "func.func(tosa-to-tensor),"
            "convert-tensor-to-linalg,"
            "one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs-from-loops=true},"
            "func.func(convert-linalg-to-loops),"
            "func.func(convert-scf-to-cf),"
            "expand-strided-metadata,"
            "lower-affine,"
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
def build_ir(input1_type, input2_type, result_type):
    input1_tensor = f"tensor<{input1_type}>"
    input2_tensor = f"tensor<{input2_type}>"
    result_tensor = f"tensor<{result_type}>"
    
    # emit_c_interface needed for ExecutionEngine to execute    
    ir = f"""
func.func @test_minimum(%arg0: {input1_tensor}, %arg1: {input2_tensor}) -> {result_tensor} attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.minimum"(%arg0, %arg1) : ({input1_tensor}, {input2_tensor}) -> {result_tensor}
  return %0 : {result_tensor}
}}
"""
    return ir

# Main function which calls all the other function to lower the calc.minimum op to llvm IR and executes them.
def run_minimum(input1, input2, input1_type, input2_type, result_type):
    
    # lowers calc to tosa
    calc_ir = build_ir(input1_type, input2_type, result_type)

    # parsing the result type to get the output shape and dtype for initializing the result memref descriptor. 
    output_shape = []
    for i in result_type.split("x")[:-1]:
        output_shape.append(int(i))
    output_type = input1.numpy().dtype

    # Initializing a result array to store the output
    result = np.zeros(output_shape, dtype=output_type)
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
# Broadcast test cases — different rank inputs to trigger calc-broadcast pass
MINIMUM_BROADCAST_TEST_CASES = [
    # Test 1: 1D vs 2D f32
    (
        torch.ones(1, 4, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32) * 2,
        "1x4xf32", "1xf32", "1x4xf32"
    ),
    # Test 2: 1D vs 3D f32
    (
        torch.randint(1, 10, (2, 3, 4), dtype=torch.float32),
        torch.randint(1, 10, (4,), dtype=torch.float32),
        "2x3x4xf32", "4xf32", "2x3x4xf32"
    ),
    # Test 3: 2D vs 3D i32
    (
        torch.randint(0, 10, (2, 3, 4), dtype=torch.int32),
        torch.randint(0, 10, (3, 4), dtype=torch.int32),
        "2x3x4xi32", "3x4xi32", "2x3x4xi32"
    ),
    # Test 4: same rank — no reshape triggered
    (
        torch.randn(4, 4, dtype=torch.float32),
        torch.randn(1, 4, dtype=torch.float32),
        "4x4xf32", "1x4xf32", "4x4xf32"
    ),
]

@pytest.mark.parametrize("input1, input2, input1_type, input2_type, result_type", MINIMUM_BROADCAST_TEST_CASES)
def test_minimum(input1, input2, input1_type, input2_type, result_type):
    
    # print the input tensors 
    print(f"\n input1: {input1}")
    print(f" input2: {input2}")

    # actual and calc outputs 
    expected = torch.minimum(input1, input2).numpy()
    actual = run_minimum(input1, input2, input1_type, input2_type, result_type)

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)
    # verify lowering didnt fail
    assert actual is not None, "run_minimum returned None — lowering failed"
    
    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)