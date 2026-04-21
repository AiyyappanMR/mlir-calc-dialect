import gc
import sys
import ctypes
import subprocess
import pytest
import torch
import numpy as np

# Import path to mlir python bindings
MLIR_CORE_PATH = "/home/mcw/llvm-project/build/tools/mlir/python_packages/mlir_core"
if MLIR_CORE_PATH not in sys.path:
    sys.path.insert(0, MLIR_CORE_PATH)
 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy

#---------------------------------------------------------------------------------
#                 Calc.logaddexp2 execution
#---------------------------------------------------------------------------------

# pass pipeline to lower tosa to llvm IR
def CalcToLLVM(module):
    pm = PassManager.parse(
        "builtin.module("
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

# Using subprocess to lower the calc.logaddexp2 to tosa dialect 
def calc_lower(calc_ir):

    # opt and lowering flag need to lower the IR
    calc_opt_path = "/home/mcw/Desktop/mlir-calc-dialect/build/bin/calc-opt"
    calc_flag = "--calc-to-tosa"

     # spawn calc-opt process and pipe the IR through stdin
    process=subprocess.Popen([calc_opt_path, calc_flag], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
    stdout, stderr = process.communicate(input = calc_ir)

    # return None if calc-opt failed
    if process.returncode != 0:
        print(f"Error in calc-opt: {stderr}", file = sys.stderr)
        return None      
    return stdout

# calc.logaddexp2 IR template which takes the tensor type and shape from the argument
def build_ir(tensor_type):
    tensor_type = f"tensor<{tensor_type}>"
    
    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_logaddexp2(%arg0: {tensor_type}, %arg1: {tensor_type}) -> {tensor_type} attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.logaddexp2"(%arg0, %arg1) : ({tensor_type}, {tensor_type}) -> {tensor_type}
  return %0 : {tensor_type}
}}
"""
    return ir

# Main function which calls all the other function to lower the calc.logaddexp2 op to llvm IR and executes them.
def run_logaddexp2(input1, input2, tensor_type):
    
    # lowers calc to tosa
    calc_ir = build_ir(tensor_type)
    lowered_ir = calc_lower(calc_ir)

    # Initializing a result array to store the output
    result = np.zeros_like(input1.numpy())
    if lowered_ir:
        with Context() as ctx:
            
            # parse the lowered IR and lower it all the way to llvm IR
            module = Module.parse(lowered_ir)
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
            execution_engine.invoke("test_logaddexp2", final_result, final_input1, final_input2)
            
            # convert the result memref back to numpy
            np_result = ranked_memref_to_numpy(ctypes.pointer(mem_result))

        return np_result
    

#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases which has the following parameter's input1, input2, and tensortype
LOGADDEXP2_TEST_CASES = [
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
    )
]

@pytest.mark.parametrize("input1, input2, tensor_type", LOGADDEXP2_TEST_CASES)
def test_logaddexp2(input1, input2, tensor_type):
    
    # actual and calc outputs 
    expected = torch.logaddexp2(input1, input2).numpy()
    actual = run_logaddexp2(input1, input2, tensor_type)

    print(f"\n[{tensor_type}] actual  : {actual}")
    print(f"[{tensor_type}] expected: {expected}")

    # verify lowering didnt fail
    assert actual is not None, "run_logaddexp2 returned None — lowering failed"
    
    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
