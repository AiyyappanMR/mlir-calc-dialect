# This file contains pytest tests for functional testing of split op with respect to pytorch's split operator.

import sys
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
import os
import calc_mlir
from calc_mlir.ir import Context, Module
from calc_mlir.passmanager import PassManager
from calc_mlir.execution_engine import ExecutionEngine
from calc_mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
from calc_mlir._mlir_libs._calcMlir import register_dialect

#---------------------------------------------------------------------------------
#                 Calc.split execution
#---------------------------------------------------------------------------------

# pass pipeline to lower calc dialect to llvm IR
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


# mapping from mlir element type strings to numpy dtypes
Num_Dty = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}


# parses an mlir type string like "4x6xf32" into shape (4, 6) and element type "f32"
def parse_mlir_type(type_str):
    parts = type_str.split("x")
    elem_type = parts[-1]
    shape = tuple(int(d) for d in parts[:-1])
    return shape, elem_type


# calc.split IR template which takes the input type, output types, split sizes, and optional dim
def build_ir(in_type, out_types, split_sizes, dim=None):
    if dim is not None:
        dim_attr = f", dim = {dim} : si32"
    else:
        dim_attr = ""
        
    sizes_str = ", ".join(map(str, split_sizes))
    out_types_str = ", ".join([f"tensor<{t}>" for t in out_types])
    
    ret_vars = ", ".join([f"%{i}" for i in range(len(out_types))])
    
    # emit_c_interface needed for ExecutionEngine to execute
    return f"""
func.func @test_split(%arg0: tensor<{in_type}>) -> ({out_types_str}) attributes {{llvm.emit_c_interface}} {{
  {ret_vars} = "calc.split"(%arg0) {{split_sizes = array<i64: {sizes_str}>{dim_attr}}} : (tensor<{in_type}>) -> ({out_types_str})
  return {ret_vars} : {out_types_str}
}}
"""


# Main function which calls all the other functions to lower the calc.split op to llvm IR and executes them.
def run_split(input_tensor, in_type, out_types, split_sizes, dim=None):
    _, elem_type = parse_mlir_type(in_type)
    np_dtype = Num_Dty[elem_type]

    # lowers calc to tosa
    calc_ir = build_ir(in_type, out_types, split_sizes, dim)
    np_input = input_tensor.numpy().astype(np_dtype)

    with Context() as ctx:
        register_dialect(ctx)

        # parse the lowered IR and lower it all the way to llvm IR
        module = Module.parse(calc_ir)
        execution_engine = ExecutionEngine(CalcToLLVM(module), shared_libs=[LIBMLIR_RUNNER_UTILS])

        # Converts the torch -> numpy -> memref
        mem_input = get_ranked_memref_descriptor(np_input)
        final_input = ctypes.pointer(ctypes.pointer(mem_input))

        # allocate output buffers and wrap them as memref descriptors
        memrefs = []
        for out_type in out_types:
            shape, _ = parse_mlir_type(out_type)
            res_np = np.zeros(shape, dtype=np_dtype)
            memrefs.append(get_ranked_memref_descriptor(res_np))
            
        # Dynamically create the result struct type based on outputs
        # refer to https://discourse.llvm.org/t/runjit-a-model-with-multiple-results/60402 for how to return 
        # multiple memrefs from the execution engine
        class ResultStruct(ctypes.Structure):
            _fields_ = [(f"res{i}", type(memrefs[i])) for i in range(len(memrefs))]
            
        result_struct = ResultStruct(*memrefs)
        final_result = ctypes.pointer(ctypes.pointer(result_struct))

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke("test_split", final_result, final_input)

        # convert the result memrefs back to numpy
        results = []
        for i in range(len(memrefs)):
            res_memref = getattr(result_struct, f"res{i}")
            results.append(ranked_memref_to_numpy(ctypes.pointer(res_memref)))

        return results


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------

# parametrized test cases with parameters: input_tensor, in_type, out_types, split_sizes, and dim
SPLIT_TEST_CASES = [
    # 1D tests
    pytest.param(
        torch.randn(4, dtype=torch.float32),
        "4xf32", ["2xf32", "2xf32"],
        [2, 2], None,
        id="1d_f32_equal_nodim",
    ),
    pytest.param(
        torch.randn(6, dtype=torch.float64),
        "6xf64", ["1xf64", "2xf64", "3xf64"],
        [1, 2, 3], None,
        id="1d_f64_3way_nodim",
    ),
    # 2D tests
    pytest.param(
        torch.randn(4, 6, dtype=torch.float32),
        "4x6xf32", ["1x6xf32", "3x6xf32"],
        [1, 3], 0,
        id="2d_f32_dim0",
    ),
    pytest.param(
        torch.randn(3, 6, dtype=torch.float32),
        "3x6xf32", ["3x2xf32", "3x4xf32"],
        [2, 4], 1,
        id="2d_f32_dim1",
    ),
    pytest.param(
        torch.randn(3, 6, dtype=torch.float64),
        "3x6xf64", ["3x2xf64", "3x4xf64"],
        [2, 4], -1,
        id="2d_f64_negdim1",
    ),
    pytest.param(
        torch.randn(3, 6, dtype=torch.float64),
        "3x6xf64", ["3x1xf64", "3x2xf64", "3x3xf64"],
        [1, 2, 3], 1,
        id="2d_f64_3way_dim1",
    ),
    # 3D tests
    pytest.param(
        torch.randn(6, 3, 4, dtype=torch.float32),
        "6x3x4xf32", ["2x3x4xf32", "4x3x4xf32"],
        [2, 4], 0,
        id="3d_f32_dim0",
    ),
    pytest.param(
        torch.randn(2, 3, 6, dtype=torch.float32),
        "2x3x6xf32", ["2x3x2xf32", "2x3x4xf32"],
        [2, 4], 2,
        id="3d_f32_dim2",
    ),
    pytest.param(
        torch.randn(2, 6, 4, dtype=torch.float64),
        "2x6x4xf64", ["2x2x4xf64", "2x4x4xf64"],
        [2, 4], -2,
        id="3d_f64_negdim2",
    ),
    pytest.param(
        torch.randn(2, 6, 4, dtype=torch.float32),
        "2x6x4xf32", ["2x1x4xf32", "2x2x4xf32", "2x3x4xf32"],
        [1, 2, 3], 1,
        id="3d_f32_3way_dim1",
    ),
    pytest.param(
        torch.randn(6, 3, 4, dtype=torch.float32),
        "6x3x4xf32", ["2x3x4xf32", "4x3x4xf32"],
        [2, 4], -3,
        id="3d_f32_negdim3",
    ),
    pytest.param(
        torch.randn(6, 3, 4, dtype=torch.float32),
        "6x3x4xf32", ["2x3x4xf32", "4x3x4xf32"],
        [2, 4], None,
        id="3d_f32_nodim",
    ),
]


@pytest.mark.parametrize("input_tensor, in_type, out_types, split_sizes, dim", SPLIT_TEST_CASES)
def test_split(input_tensor, in_type, out_types, split_sizes, dim):
    _, elem_type = parse_mlir_type(in_type)
    print("-" * 80)

    # actual and expected outputs
    actuals = run_split(input_tensor, in_type, out_types, split_sizes, dim)

    if dim is None:
        expected = torch.split(input_tensor, split_sizes)
    else:
        expected = torch.split(input_tensor, split_sizes, dim=dim)

    for actual, exp in zip(actuals, expected):
        exp_np = exp.numpy()
        print(f"\n calc output : {actual}")
        print(f" torch output : {exp_np}")

        # verify lowering didn't fail
        assert actual is not None, f"run_split returned None for output"

        # compare actual vs expected element-wise with tolerance for floating point operations
        if actual.dtype in (np.float32, np.float64):
            np.testing.assert_allclose(actual, exp_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(actual, exp_np)
            
    print("-" * 80)
