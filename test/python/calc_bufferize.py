# This file contains pytest tests for functional testing of calc.stack op with respect to pytorch's torch.stack operator.

import sys
import ctypes
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
#                 Calc.stack execution
#---------------------------------------------------------------------------------

def CalcBufferizeToLLVM(module):
    pm = PassManager.parse(
        "builtin.module("
            "calc-bufferize,"
            "one-shot-bufferize{bufferize-function-boundaries=1},"
            "calc-to-memref,"
            "expand-strided-metadata,"
            "finalize-memref-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


# calc.stack IR template which takes the tensor type, result type, number of inputs, and dim from the argument
def build_ir(tensor_type, result_type, num_inputs, dim):
    tensor_type_str = f"tensor<{tensor_type}>"
    result_type_str = f"tensor<{result_type}>"

    # build function args: %arg0, %arg1, ... based on num_inputs
    args      = ", ".join([f"%arg{i}: {tensor_type_str}" for i in range(num_inputs)])
    operands  = ", ".join([f"%arg{i}" for i in range(num_inputs)])
    operand_types = ", ".join([tensor_type_str] * num_inputs)

    # dim attr is optional — omit it if None to test default dim=0 behavior
    dim_attr = f"{{dim = {dim} : si32}}" if dim is not None else ""

    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_stack({args}) -> {result_type_str} attributes {{llvm.emit_c_interface}} {{
  %0 = calc.stack {operands} {dim_attr} : ({operand_types}) -> {result_type_str}
  return %0 : {result_type_str}
}}
"""
    return ir


# Main function which calls all the other functions to lower the calc.stack op to llvm IR and executes them.
def run_stack(input_tensors, tensor_type, result_type, dim):

    num_inputs = len(input_tensors)

    # lowers calc to tosa
    calc_ir = build_ir(tensor_type, result_type, num_inputs, dim)

    # get result shape from torch to initialize the result array
    torch_dim = dim if dim is not None else 0
    torch_result = torch.stack(input_tensors, dim=torch_dim)
    result = np.zeros_like(torch_result.numpy())

    with Context() as ctx:

        register_dialect(ctx)

        # parse the lowered IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(
            CalcBufferizeToLLVM(module),
            shared_libs=["/home/mcw/llvm-project/build/lib/libmlir_c_runner_utils.so"],
        )

        # build memref descriptors for each input tensor
        mem_inputs = [get_ranked_memref_descriptor(t.numpy()) for t in input_tensors]
        mem_result = get_ranked_memref_descriptor(result)

        # pointer to the memref struct instead of value
        final_inputs = [ctypes.pointer(ctypes.pointer(m)) for m in mem_inputs]
        final_result = ctypes.pointer(ctypes.pointer(mem_result))

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_stack", final_result, *final_inputs)

        # convert the result memref back to numpy
        return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases mirroring the LIT tests:
# 1d/2d/3d/4d x i32/i64/f32/f64 x 1/2/3/4 inputs x dim=0/1/2/neg/None
STACK_TEST_CASES = [
    # --- 1D inputs ---

    # 1D i32, 1 input, dim=0
    (
        [torch.tensor([1, 2, 3, 4], dtype=torch.int32)],
        "4xi32", "1x4xi32", 0,
    ),
    # 1D f32, 2 inputs, dim=0
    (
        [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
         torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)],
        "4xf32", "2x4xf32", 0,
    ),
    # 1D i64, 3 inputs, dim=1
    (
        [torch.tensor([1, 2, 3, 4], dtype=torch.int64),
         torch.tensor([5, 6, 7, 8], dtype=torch.int64),
         torch.tensor([9, 10, 11, 12], dtype=torch.int64)],
        "4xi64", "4x3xi64", 1,
    ),
    # 1D f64, 4 inputs, neg dim=-1 (normalizes to dim=1)
    (
        [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
         torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64),
         torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64),
         torch.tensor([13.0, 14.0, 15.0, 16.0], dtype=torch.float64)],
        "4xf64", "4x4xf64", -1,
    ),

    # --- 2D inputs ---

    # 2D f32, 1 input, dim=0
    (
        [torch.ones(3, 4, dtype=torch.float32)],
        "3x4xf32", "1x3x4xf32", 0,
    ),
    # 2D i32, 2 inputs, dim=1
    (
        [torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32),
         torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)],
        "3x4xi32", "3x2x4xi32", 1,
    ),
    # 2D f64, 3 inputs, dim=2
    (
        [torch.randn(3, 4, dtype=torch.float64),
         torch.randn(3, 4, dtype=torch.float64),
         torch.randn(3, 4, dtype=torch.float64)],
        "3x4xf64", "3x4x3xf64", 2,
    ),
    # 2D i64, 4 inputs, neg dim=-1 (normalizes to dim=2)
    (
        [torch.ones(3, 4, dtype=torch.int64),
         torch.ones(3, 4, dtype=torch.int64),
         torch.ones(3, 4, dtype=torch.int64),
         torch.ones(3, 4, dtype=torch.int64)],
        "3x4xi64", "3x4x4xi64", -1,
    ),
    # 2D i32, 2 inputs, no dim attr (default dim=0)
    (
        [torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
         torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)],
        "2x2xi32", "2x2x2xi32", None,
    ),

    # --- 3D inputs ---

    # 3D i32, 1 input, dim=0
    (
        [torch.ones(2, 3, 4, dtype=torch.int32)],
        "2x3x4xi32", "1x2x3x4xi32", 0,
    ),
    # 3D f32, 2 inputs, dim=1
    (
        [torch.randn(2, 3, 4, dtype=torch.float32),
         torch.randn(2, 3, 4, dtype=torch.float32)],
        "2x3x4xf32", "2x2x3x4xf32", 1,
    ),
    # 3D i64, 3 inputs, dim=2
    (
        [torch.ones(2, 3, 4, dtype=torch.int64),
         torch.ones(2, 3, 4, dtype=torch.int64),
         torch.ones(2, 3, 4, dtype=torch.int64)],
        "2x3x4xi64", "2x3x3x4xi64", 2,
    ),
    # 3D f64, 4 inputs, dim=3
    (
        [torch.randn(2, 3, 4, dtype=torch.float64),
         torch.randn(2, 3, 4, dtype=torch.float64),
         torch.randn(2, 3, 4, dtype=torch.float64),
         torch.randn(2, 3, 4, dtype=torch.float64)],
        "2x3x4xf64", "2x3x4x4xf64", 3,
    ),
    # 3D f32, 2 inputs, neg dim=-2 (normalizes to dim=2)
    (
        [torch.randn(2, 3, 4, dtype=torch.float32),
         torch.randn(2, 3, 4, dtype=torch.float32)],
        "2x3x4xf32", "2x3x2x4xf32", -2,
    ),

    # --- 4D inputs ---

    # 4D f64, 1 input, dim=2
    (
        [torch.randn(2, 3, 4, 5, dtype=torch.float64)],
        "2x3x4x5xf64", "2x3x1x4x5xf64", 2,
    ),
    # 4D i32, 2 inputs, dim=1
    (
        [torch.ones(2, 3, 4, 5, dtype=torch.int32),
         torch.ones(2, 3, 4, 5, dtype=torch.int32)],
        "2x3x4x5xi32", "2x2x3x4x5xi32", 1,
    ),
    # 4D f32, 3 inputs, dim=4 (append at end)
    (
        [torch.randn(2, 3, 4, 5, dtype=torch.float32),
         torch.randn(2, 3, 4, 5, dtype=torch.float32),
         torch.randn(2, 3, 4, 5, dtype=torch.float32)],
        "2x3x4x5xf32", "2x3x4x5x3xf32", 4,
    ),
    # 4D i64, 4 inputs, dim=0
    (
        [torch.ones(2, 3, 4, 5, dtype=torch.int64),
         torch.ones(2, 3, 4, 5, dtype=torch.int64),
         torch.ones(2, 3, 4, 5, dtype=torch.int64),
         torch.ones(2, 3, 4, 5, dtype=torch.int64)],
        "2x3x4x5xi64", "4x2x3x4x5xi64", 0,
    ),
]


@pytest.mark.parametrize("input_tensors, tensor_type, result_type, dim", STACK_TEST_CASES)
def test_stack(input_tensors, tensor_type, result_type, dim):

    # actual and expected outputs
    actual = run_stack(input_tensors, tensor_type, result_type, dim)

    torch_dim = dim if dim is not None else 0
    expected = torch.stack(input_tensors, dim=torch_dim).numpy()

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)

    # verify lowering didn't fail
    assert actual is not None, "run_stack returned None — lowering failed"

    # compare actual vs expected element-wise with a tolerance for floating point types
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)