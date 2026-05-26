# This file contains pytest tests for functional testing of softmax op with respect to pytorch's softmax operator.

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
#                 Calc.softmax execution
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


# calc.softmax IR template which takes the tensor type, shape, and dim from the argument
def build_ir(tensor_type, dim):
    tensor_type = f"tensor<{tensor_type}>"

    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_softmax(%arg0: {tensor_type}) -> {tensor_type} attributes {{llvm.emit_c_interface}} {{
  %0 = calc.softmax %arg0 {{dim = {dim} : si32}} : ({tensor_type}) -> {tensor_type}
  return %0 : {tensor_type}
}}
"""
    return ir


# Main function which calls all the other functions to lower the calc.softmax op to llvm IR and executes them.
def run_softmax(input_tensor, tensor_type, dim):

    # lowers calc to tosa
    calc_ir = build_ir(tensor_type, dim)

    # Initializing a result array to store the output
    result = np.zeros_like(input_tensor.numpy())

    with Context() as ctx:

        register_dialect(ctx)

        # parse the lowered IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(
            CalcToLLVM(module),
            shared_libs=["/home/mcw/llvm-project/build/lib/libmlir_c_runner_utils.so"],
        )

        # Converts the torch -> numpy -> memref
        mem_input = get_ranked_memref_descriptor(input_tensor.numpy())
        mem_result = get_ranked_memref_descriptor(result)

        # pointer to the memref struct instead of value
        final_input = ctypes.pointer(ctypes.pointer(mem_input))
        final_result = ctypes.pointer(ctypes.pointer(mem_result))

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_softmax", final_result, final_input)

        # convert the result memref back to numpy
        return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases which has the following parameters: input_tensor, tensor_type, and dim
SOFTMAX_TEST_CASES = [
    (
        torch.randn(4, 4, dtype=torch.float32),
        "4x4xf32",
        0,
    ),
    (
        torch.randn(4, 4, dtype=torch.float32),
        "4x4xf32",
        1,
    ),
    (
        torch.randn(4, 4, dtype=torch.float32),
        "4x4xf32",
        -1,
    ),
    (
        torch.randn(2, 4, 8, dtype=torch.float32),
        "2x4x8xf32",
        -2,
    ),
    (
        torch.randn(8, dtype=torch.float32),
        "8xf32",
        0,
    ),
    (
        torch.randn(2, 4, 8, dtype=torch.float64),
        "2x4x8xf64",
        2,
    ),
    (
        torch.randn(2, 8, 16, 16, dtype=torch.float64),
        "2x8x16x16xf64",
        3,
    ),
]


@pytest.mark.parametrize("input_tensor, tensor_type, dim", SOFTMAX_TEST_CASES)
def test_softmax(input_tensor, tensor_type, dim):

    # print the input tensor
    print(f"\n input: {input_tensor}")
    print(f" tensor_type: {tensor_type}, dim: {dim}")

    # actual and expected outputs
    actual = run_softmax(input_tensor, tensor_type, dim)

    sm = torch.nn.Softmax(dim=dim)
    expected = sm(input_tensor).numpy()

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)

    # verify lowering didn't fail
    assert actual is not None, "run_softmax returned None — lowering failed"

    # compare actual vs expected element-wise with a tolerance since we are doing floating point operations
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)