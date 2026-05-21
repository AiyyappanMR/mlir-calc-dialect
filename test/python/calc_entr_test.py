# This file contains pytest tests for functional testing of entr op with respect to
# the reference definition: entr(x) = -x*log(x) for x>0, 0 for x==0, -inf for x<0.

import sys
import ctypes
import pytest
import torch
import numpy as np
import torch

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
#                 Calc.entr execution
#---------------------------------------------------------------------------------

# pass pipeline to lower tosa to llvm IR
def CalcToLLVM(module):
    pm = PassManager.parse(
        "builtin.module("
            "calc-to-tosa,"
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


# build_ir takes separate input and result types because integer inputs produce float results.
def build_ir(input_type, result_type):
    ir = f"""
func.func @test_entr(%arg0: tensor<{input_type}>) -> tensor<{result_type}> attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.entr"(%arg0) : (tensor<{input_type}>) -> tensor<{result_type}>
  return %0 : tensor<{result_type}>
}}
"""
    return ir

# Main function which calls all the other function to lower the calc.entr op to llvm IR and executes them.
def run_entr(input_tensor, input_type, result_type):
    
    # lowers calc to tosa
    calc_ir = build_ir(input_type, result_type)

    # Initializing a result array to store the output
    result_dtype = np.float64 if "f64" in result_type else np.float32
    result = np.zeros(input_tensor.numpy().shape, dtype=result_dtype)

    with Context() as ctx:

        register_dialect(ctx)
        
        # parse the lowered IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(CalcToLLVM(module))

        # Converts the torch -> numpy -> memref
        mem_input = get_ranked_memref_descriptor(input_tensor.numpy())
        mem_result = get_ranked_memref_descriptor(result)

        # pointer to the memref struct  instead of value
        final_input = ctypes.pointer(ctypes.pointer(mem_input))
        final_result = ctypes.pointer(ctypes.pointer(mem_result))

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_entr", final_result, final_input)

        # convert the result memref back to numpy
        return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------

# parametrized test cases which has the following parameter's input1 value and input/output tensortype
ENTR_TEST_CASES = [
    # --- f32 cases ---
    pytest.param(
        torch.tensor([0.5, 0.0, -1.0, 1.0], dtype=torch.float32),
        "4xf32", "4xf32",
        id="1d_f32_mixed_pos_zero_neg",
    ),
    pytest.param(
        torch.rand(6, dtype=torch.float32),
        "6xf32", "6xf32",
        id="1d_f32_random_positive",
    ),
    pytest.param(
        torch.zeros(5, dtype=torch.float32),
        "5xf32", "5xf32",
        id="1d_f32_all_zeros",
    ),
    pytest.param(
        torch.tensor([-0.5, -1.0, -2.0], dtype=torch.float32),
        "3xf32", "3xf32",
        id="1d_f32_all_negative",
    ),
    pytest.param(
        torch.randn(3, 3, dtype=torch.float32).abs(),
        "3x3xf32", "3x3xf32",
        id="2d_f32_positive",
    ),
    pytest.param(
        torch.randn(2, 2, 3, dtype=torch.float32),
        "2x2x3xf32", "2x2x3xf32",
        id="3d_f32_mixed",
    ),
    # --- f64 cases ---
    pytest.param(
        torch.tensor([0.5, 0.0, -1.0, 1.0], dtype=torch.float64),
        "4xf64", "4xf64",
        id="1d_f64_mixed_pos_zero_neg",
    ),
    pytest.param(
        torch.rand(6, dtype=torch.float64),
        "6xf64", "6xf64",
        id="1d_f64_random_positive",
    ),
    pytest.param(
        torch.randn(2, 4, dtype=torch.float64),
        "2x4xf64", "2x4xf64",
        id="2d_f64_mixed",
    ),
    # --- integer input cases (entr converts to float internally) ---
    pytest.param(
        torch.tensor([0, 1, 2, -1], dtype=torch.int32),
        "4xi32", "4xf32",
        id="1d_i32_to_f32",
    ),
    pytest.param(
        torch.tensor([0, 1, 3, -2], dtype=torch.int64),
        "4xi64", "4xf64",
        id="1d_i64_to_f64",
    ),
]

@pytest.mark.parametrize("input_tensor, input_type, result_type", ENTR_TEST_CASES)
def test_entr(input_tensor, input_type, result_type):
    
    # actual and calc outputs 
    expected = torch.special.entr(input_tensor).numpy()
    actual = run_entr(input_tensor, input_type, result_type)

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)

    # verify lowering didnt fail
    assert actual is not None, "run_entr returned None — lowering failed"

    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
