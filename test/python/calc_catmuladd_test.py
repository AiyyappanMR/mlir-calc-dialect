# This file contains pytest tests for functional testing of catmuladd op with respect to pytorch's
# torch.cat + torch.mul (+ torch.add) operations.

import sys
import os
import ctypes
import pytest
import torch
import numpy as np

# calc_mlir is a self-contained package that bundles MLIR core + our dialect.
CALC_PYTHON_PATH = "/home/mcw/Desktop/mlir-calc-dialect/build/python_packages/calc_mlir"
if CALC_PYTHON_PATH not in sys.path:
    sys.path.insert(0, CALC_PYTHON_PATH)

LIBMLIR_RUNNER_UTILS = "/home/mcw/llvm-project/build/lib/libmlir_c_runner_utils.so"

# Importing the necessary components from calc_mlir to build, lower, and execute the IR.
import calc_mlir
from calc_mlir.ir import Context, Module
from calc_mlir.passmanager import PassManager
from calc_mlir.execution_engine import ExecutionEngine
from calc_mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
from calc_mlir._mlir_libs._calcMlir import register_dialect

#---------------------------------------------------------------------------------
#                 Calc.catmuladd execution
#---------------------------------------------------------------------------------

# pass pipeline to lower calc to llvm IR
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
            "convert-arith-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module

def build_ir(inputs1_types, inputs2_types, result_type, scale_type=None, dim=None):
    n1 = len(inputs1_types)
    n2 = len(inputs2_types)

    # build arg list: inputs1 args, then inputs2 args, then optional scale arg
    args = []
    for i, t in enumerate(inputs1_types):
        args.append(f"%arg{i}: tensor<{t}>")
    for i, t in enumerate(inputs2_types):
        args.append(f"%arg{n1 + i}: tensor<{t}>")
    if scale_type is not None:
        args.append(f"%arg{n1 + n2}: tensor<{scale_type}>")
    args_str = ", ".join(args)

    # build the operand type list: inputs1 + inputs2 + optional scale
    operand_types = [f"tensor<{t}>" for t in inputs1_types + inputs2_types]
    if scale_type is not None:
        operand_types.append(f"tensor<{scale_type}>")
    operand_types_str = ", ".join(operand_types)

    # all operand value references in order: inputs1, inputs2, optional scale
    all_vals = [f"%arg{i}" for i in range(n1)] + [f"%arg{n1 + i}" for i in range(n2)]
    if scale_type is not None:
        all_vals.append(f"%arg{n1 + n2}")
    all_vals_str = ", ".join(all_vals)

    scale_count = 1 if scale_type is not None else 0
    seg_sizes = f"array<i32: {n1}, {n2}, {scale_count}>"

    # optional dim attribute appended into the generic attribute dict
    dim_attr = f", dim = {dim} : si32" if dim is not None else ""

    result_tensor = f"tensor<{result_type}>"

    # emit_c_interface needed for ExecutionEngine to execute
    ir = f"""
func.func @test_catmuladd({args_str}) -> {result_tensor} attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.catmuladd"({all_vals_str}) <{{operandSegmentSizes = {seg_sizes}{dim_attr}}}>  : ({operand_types_str}) -> {result_tensor}
  return %0 : {result_tensor}
}}
"""
    return ir


# Main function which calls all the other functions to lower the calc.catmuladd op to llvm IR and executes them.
# inputs1 and inputs2 are lists of torch tensors; scale is an optional torch tensor.
def run_catmuladd(inputs1, inputs2, inputs1_types, inputs2_types, result_type, scale=None, scale_type=None, dim=None):

    # lowers calc to tosa
    calc_ir = build_ir(inputs1_types, inputs2_types, result_type, scale_type=scale_type, dim=dim)

    # parse the result type to get the output shape and dtype for initializing the result memref descriptor
    parts = result_type.split("x")
    output_shape = [int(p) for p in parts[:-1]]
    output_dtype = inputs1[0].numpy().dtype

    # Initializing a result array to store the output
    result = np.zeros(output_shape, dtype=output_dtype)

    with Context() as ctx:

        register_dialect(ctx)

        # parse the IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(CalcToLLVM(module), shared_libs=[LIBMLIR_RUNNER_UTILS])

        # Converts the torch -> numpy -> memref for all variadic inputs
        mem_inputs1 = [get_ranked_memref_descriptor(t.numpy()) for t in inputs1]
        mem_inputs2 = [get_ranked_memref_descriptor(t.numpy()) for t in inputs2]
        mem_result  = get_ranked_memref_descriptor(result)

        # pointer to the memref struct instead of value
        final_inputs1 = [ctypes.pointer(ctypes.pointer(m)) for m in mem_inputs1]
        final_inputs2 = [ctypes.pointer(ctypes.pointer(m)) for m in mem_inputs2]
        final_result  = ctypes.pointer(ctypes.pointer(mem_result))

        # build invocation arg list: result first (ABI), then inputs1, then inputs2, then optional scale
        invoke_args = [final_result] + final_inputs1 + final_inputs2
        if scale is not None:
            mem_scale   = get_ranked_memref_descriptor(scale.numpy())
            final_scale = ctypes.pointer(ctypes.pointer(mem_scale))
            invoke_args.append(final_scale)

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_catmuladd", *invoke_args)

        # convert the result memref back to numpy
        return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------


# parametrized test cases which map directly to the 14 tests in test/calc/catmuladd.mlir
# each entry: (inputs1, inputs2, inputs1_types, inputs2_types, result_type, scale, scale_type, dim)
CATMULADD_TEST_CASES = [

    # Test 1: 1D f32, 1+1 inputs, dim=0, no scale
    pytest.param(
        [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)],
        [torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)],
        ["4xf32"], ["4xf32"], "4xf32",
        None, None, 0,
        id="1d_1p1inp_f32_dim0_noscale",
    ),

    # Test 2: 1D f32, 2+2 inputs, dim=0, no scale
    pytest.param(
        [torch.tensor([1.0, 2.0], dtype=torch.float32),
         torch.tensor([3.0, 4.0], dtype=torch.float32)],
        [torch.tensor([5.0, 6.0], dtype=torch.float32),
         torch.tensor([7.0, 8.0], dtype=torch.float32)],
        ["2xf32", "2xf32"], ["2xf32", "2xf32"], "4xf32",
        None, None, 0,
        id="1d_2p2inp_f32_dim0_noscale",
    ),

    # Test 3: 1D i32, 2+2 inputs, dim=0, with scale
    pytest.param(
        [torch.tensor([1, 2], dtype=torch.int32),
         torch.tensor([3, 4], dtype=torch.int32)],
        [torch.tensor([5, 6], dtype=torch.int32),
         torch.tensor([7, 8], dtype=torch.int32)],
        ["2xi32", "2xi32"], ["2xi32", "2xi32"], "4xi32",
        torch.tensor([10, 20, 30, 40], dtype=torch.int32), "4xi32", 0,
        id="1d_2p2inp_i32_dim0_scale",
    ),

    # Test 4: 1D f64, 3+3 inputs, no dim attr, no scale
    pytest.param(
        [torch.tensor([1.0, 2.0], dtype=torch.float64),
         torch.tensor([3.0, 4.0], dtype=torch.float64),
         torch.tensor([5.0, 6.0], dtype=torch.float64)],
        [torch.tensor([7.0, 8.0], dtype=torch.float64),
         torch.tensor([9.0, 10.0], dtype=torch.float64),
         torch.tensor([11.0, 12.0], dtype=torch.float64)],
        ["2xf64", "2xf64", "2xf64"], ["2xf64", "2xf64", "2xf64"], "6xf64",
        None, None, None,
        id="1d_3p3inp_f64_nodim_noscale",
    ),

    # Test 5: 2D f32, 2+2 inputs, dim=0, no scale
    pytest.param(
        [torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], dtype=torch.float32)],
        [torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=torch.float32),
         torch.tensor([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=torch.float32)],
        ["2x4xf32", "2x4xf32"], ["2x4xf32", "2x4xf32"], "4x4xf32",
        None, None, 0,
        id="2d_2p2inp_f32_dim0_noscale",
    ),

    # Test 6: 2D i32, 2+2 inputs, dim=1, no scale
    pytest.param(
        [torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
         torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32)],
        [torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.int32),
         torch.tensor([[2, 2], [2, 2], [2, 2]], dtype=torch.int32)],
        ["3x2xi32", "3x2xi32"], ["3x2xi32", "3x2xi32"], "3x4xi32",
        None, None, 1,
        id="2d_2p2inp_i32_dim1_noscale",
    ),

    # Test 7: 2D f64, 2+2 inputs, neg dim=-1 (==dim=1), no scale
    pytest.param(
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64),
         torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float64)],
        [torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
         torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=torch.float64)],
        ["3x2xf64", "3x2xf64"], ["3x2xf64", "3x2xf64"], "3x4xf64",
        None, None, -1,
        id="2d_2p2inp_f64_negdim1_noscale",
    ),

    # Test 8: 2D f32, 2+2 inputs, dim=1, with scale
    pytest.param(
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
         torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)],
        [torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
         torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)],
        ["3x2xf32", "3x2xf32"], ["3x2xf32", "3x2xf32"], "3x4xf32",
        torch.ones(3, 4, dtype=torch.float32) * 0.5, "3x4xf32", 1,
        id="2d_2p2inp_f32_dim1_scale",
    ),

    # Test 9: 2D i64, 3+3 inputs, dim=0, with broadcast scale
    pytest.param(
        [torch.ones(2, 4, dtype=torch.int64),
         torch.ones(2, 4, dtype=torch.int64) * 2,
         torch.ones(2, 4, dtype=torch.int64) * 3],
        [torch.ones(2, 4, dtype=torch.int64),
         torch.ones(2, 4, dtype=torch.int64),
         torch.ones(2, 4, dtype=torch.int64)],
        ["2x4xi64", "2x4xi64", "2x4xi64"], ["2x4xi64", "2x4xi64", "2x4xi64"], "6x4xi64",
        torch.ones(1, 4, dtype=torch.int64) * 10, "1x4xi64", 0,
        id="2d_3p3inp_i64_dim0_bcastscale",
    ),

    # Test 10: 3D f32, 2+2 inputs, dim=0, no scale
    pytest.param(
        [torch.ones(2, 3, 4, dtype=torch.float32),
         torch.ones(2, 3, 4, dtype=torch.float32) * 2.0],
        [torch.ones(2, 3, 4, dtype=torch.float32) * 3.0,
         torch.ones(2, 3, 4, dtype=torch.float32) * 4.0],
        ["2x3x4xf32", "2x3x4xf32"], ["2x3x4xf32", "2x3x4xf32"], "4x3x4xf32",
        None, None, 0,
        id="3d_2p2inp_f32_dim0_noscale",
    ),

    # Test 11: 3D i32, 2+2 inputs, dim=2, no scale
    pytest.param(
        [torch.ones(2, 3, 2, dtype=torch.int32),
         torch.ones(2, 3, 2, dtype=torch.int32) * 2],
        [torch.ones(2, 3, 2, dtype=torch.int32) * 3,
         torch.ones(2, 3, 2, dtype=torch.int32) * 4],
        ["2x3x2xi32", "2x3x2xi32"], ["2x3x2xi32", "2x3x2xi32"], "2x3x4xi32",
        None, None, 2,
        id="3d_2p2inp_i32_dim2_noscale",
    ),

    # Test 12: 3D f64, 2+2 inputs, neg dim=-2 (==dim=1), no scale
    pytest.param(
        [torch.ones(2, 2, 4, dtype=torch.float64),
         torch.ones(2, 2, 4, dtype=torch.float64) * 2.0],
        [torch.ones(2, 2, 4, dtype=torch.float64) * 3.0,
         torch.ones(2, 2, 4, dtype=torch.float64) * 4.0],
        ["2x2x4xf64", "2x2x4xf64"], ["2x2x4xf64", "2x2x4xf64"], "2x4x4xf64",
        None, None, -2,
        id="3d_2p2inp_f64_negdim2_noscale",
    ),

    # Test 13: 3D f32, 3+3 inputs, dim=1, with scale
    pytest.param(
        [torch.ones(2, 2, 4, dtype=torch.float32),
         torch.ones(2, 2, 4, dtype=torch.float32) * 2.0,
         torch.ones(2, 2, 4, dtype=torch.float32) * 3.0],
        [torch.ones(2, 2, 4, dtype=torch.float32),
         torch.ones(2, 2, 4, dtype=torch.float32),
         torch.ones(2, 2, 4, dtype=torch.float32)],
        ["2x2x4xf32", "2x2x4xf32", "2x2x4xf32"], ["2x2x4xf32", "2x2x4xf32", "2x2x4xf32"], "2x6x4xf32",
        torch.ones(2, 6, 4, dtype=torch.float32) * 0.5, "2x6x4xf32", 1,
        id="3d_3p3inp_f32_dim1_scale",
    ),

    # Test 14: 3D i64, 4+4 inputs, dim=2, with broadcast scale
    pytest.param(
        [torch.ones(2, 3, 2, dtype=torch.int64),
         torch.ones(2, 3, 2, dtype=torch.int64) * 2,
         torch.ones(2, 3, 2, dtype=torch.int64) * 3,
         torch.ones(2, 3, 2, dtype=torch.int64) * 4],
        [torch.ones(2, 3, 2, dtype=torch.int64),
         torch.ones(2, 3, 2, dtype=torch.int64),
         torch.ones(2, 3, 2, dtype=torch.int64),
         torch.ones(2, 3, 2, dtype=torch.int64)],
        ["2x3x2xi64", "2x3x2xi64", "2x3x2xi64", "2x3x2xi64"],
        ["2x3x2xi64", "2x3x2xi64", "2x3x2xi64", "2x3x2xi64"], "2x3x8xi64",
        torch.ones(1, 1, 8, dtype=torch.int64) * 5, "1x1x8xi64", 2,
        id="3d_4p4inp_i64_dim2_bcastscale",
    ),

    # Test 15: 2D f32, unequal number of operands (3+2), dim=0, no scale
    pytest.param(
        [torch.ones(2, 4, dtype=torch.float32) * 1.0,
         torch.ones(2, 4, dtype=torch.float32) * 2.0,
         torch.ones(2, 4, dtype=torch.float32) * 3.0],
        [torch.ones(3, 4, dtype=torch.float32) * 4.0,
         torch.ones(3, 4, dtype=torch.float32) * 5.0],
        ["2x4xf32", "2x4xf32", "2x4xf32"], ["3x4xf32", "3x4xf32"], "6x4xf32",
        None, None, 0,
        id="2d_3p2inp_f32_unequal_ops_dim0",
    ),
]


@pytest.mark.parametrize(
    "inputs1, inputs2, inputs1_types, inputs2_types, result_type, scale, scale_type, dim",
    CATMULADD_TEST_CASES,
)
def test_catmuladd(inputs1, inputs2, inputs1_types, inputs2_types, result_type, scale, scale_type, dim):

    # actual output from the calc dialect lowering
    actual = run_catmuladd(
        inputs1, inputs2, inputs1_types, inputs2_types, result_type,
        scale=scale, scale_type=scale_type, dim=dim,
    )

    # expected output: cat each group along the resolved dim, multiply, then optionally add scale
    rank = inputs1[0].ndim
    resolved_dim = (dim if dim is not None else 0)
    if resolved_dim < 0:
        resolved_dim = rank + resolved_dim
    cat1 = torch.cat(inputs1, dim=resolved_dim)
    cat2 = torch.cat(inputs2, dim=resolved_dim)
    expected = (cat1 * cat2)
    if scale is not None:
        expected = expected + scale
    expected = expected.numpy()

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)

    # verify lowering didn't fail
    assert actual is not None, "run_catmuladd returned None — lowering failed"

    # compare actual vs expected element-wise; use allclose for float types, array_equal for integer types
    if np.issubdtype(actual.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)