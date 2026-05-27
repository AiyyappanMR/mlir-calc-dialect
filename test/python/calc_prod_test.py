# This file contains pytest tests for functional testing of prod op with respect to pytorch's prod operator.

import sys
import ctypes
import pytest
import torch
import numpy as np

CALC_PYTHON_PATH = "/home/mcw/Desktop/mlir-calc-dialect/build/python_packages/calc_mlir"
if CALC_PYTHON_PATH not in sys.path:
    sys.path.insert(0, CALC_PYTHON_PATH)

import os
import calc_mlir
from calc_mlir.ir import Context, Module
from calc_mlir.passmanager import PassManager
from calc_mlir.execution_engine import ExecutionEngine
from calc_mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy, make_zero_d_memref_descriptor
from calc_mlir._mlir_libs._calcMlir import register_dialect

#---------------------------------------------------------------------------------
#                 Calc.prod execution
#---------------------------------------------------------------------------------

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


Num_Dty = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}


def parse_mlir_type(type_str):
    parts = type_str.split("x")
    elem_type = parts[-1]
    shape = tuple(int(d) for d in parts[:-1])
    return shape, elem_type


def build_ir(in_type, out_type, dim=None, keepdim=False):
    if dim is not None:
        if keepdim:
            kd = ", keepdim = true"
        else:
            kd = ""
        attrs = f"{{dim = {dim} : si32{kd}}}"
        return f"""
func.func @test_prod(%arg0: tensor<{in_type}>) -> tensor<{out_type}> attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.prod"(%arg0) {attrs} : (tensor<{in_type}>) -> tensor<{out_type}>
  return %0 : tensor<{out_type}>
}}
"""
    else:
        _, elem_type = parse_mlir_type(in_type)
        return f"""
func.func @test_prod(%arg0: tensor<{in_type}>) -> tensor<{out_type}> attributes {{llvm.emit_c_interface}} {{
  %0 = "calc.prod"(%arg0) : (tensor<{in_type}>) -> tensor<{elem_type}>
  %scalar = tensor.extract %0[] : tensor<{elem_type}>
  %result = tensor.from_elements %scalar : tensor<{out_type}>
  return %result : tensor<{out_type}>
}}
"""


def run_prod(input_tensor, in_type, out_type, dim=None, keepdim=False):
    _, elem_type = parse_mlir_type(in_type)
    out_shape, _ = parse_mlir_type(out_type)
    np_dtype = Num_Dty[elem_type]

    calc_ir = build_ir(in_type, out_type, dim, keepdim)
    np_input = input_tensor.numpy().astype(np_dtype)

    with Context() as ctx:
        register_dialect(ctx)
        print(calc_ir) 
        module = Module.parse(calc_ir)

        # Required for tests involving multi-dim tensors with no dim attr
        # export MLIR_C_RUNNER_UTILS=/path/to/llvm-project/build/lib/libmlir_c_runner_utils.so
        runner_utils_lib = os.environ.get("MLIR_C_RUNNER_UTILS")
        print(runner_utils_lib)
        if runner_utils_lib:
            execution_engine = ExecutionEngine(
                CalcToLLVM(module),
                shared_libs=[runner_utils_lib],
            )
        else:
            execution_engine = ExecutionEngine(CalcToLLVM(module))

        mem_input = get_ranked_memref_descriptor(np_input)

        final_input = ctypes.pointer(ctypes.pointer(mem_input))

        if dim is None:
            # rank-0 result using make_zero_d_memref_descriptor with rank=0
            Num_Cty = {"f32": ctypes.c_float, "f64": ctypes.c_double, "i32": ctypes.c_int32, "i64": ctypes.c_int64,}
            ctype_dtype = Num_Cty[elem_type]
            ZeroDDescriptor = make_zero_d_memref_descriptor(ctype_dtype)
            mem_result = ZeroDDescriptor()
            final_result = ctypes.pointer(ctypes.pointer(mem_result))
            execution_engine.invoke("test_prod", final_result, final_input)
            return np.array(mem_result.aligned[0], dtype=np_dtype)
        else:
            result = np.zeros(out_shape, dtype=np_dtype)
            mem_result = get_ranked_memref_descriptor(result)
            final_result = ctypes.pointer(ctypes.pointer(mem_result))
            execution_engine.invoke("test_prod", final_result, final_input)
            return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------

PROD_TEST_CASES = [
    # --- no dim: product of all elements → rank-0 result (tensor<T>) ---
    pytest.param(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        "4xf32", "f32",
        None, False,
        id="1d_f32_nodim",
    ),
    pytest.param(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        "2x2xf64", "f64",
        None, False,
        id="2d_f64_nodim",
    ),
    pytest.param(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        "4xi32", "i32",
        None, False,
        id="1d_i32_nodim",
    ),
    pytest.param(
        torch.tensor([[3, 2, 4], [5, 6, 7]], dtype=torch.int64),
        "2x3xi64", "i64",
        None, False,
        id="2d_i64_nodim",
    ),
    pytest.param(
        torch.randint(1,10,(4, 3, 2, 4), dtype=torch.int64),
        "4x3x2x4xi64", "i64",
        None, False,
        id="4d_i64_nodim",
    ),
    pytest.param(
        torch.randint(1,10,(5, 4, 3, 2, 4), dtype=torch.int64),
        "5x4x3x2x4xi64", "i64",
        None, False,
        id="5d_i64_nodim",
    ),
    # --- dim specified, keepdim=False: rank decreases by 1 ---
    pytest.param(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        "2x3xf32", "3xf32",
        0, False,
        id="2d_f32_dim0_nokeepdim",
    ),
    pytest.param(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        "2x3xi32", "2xi32",
        1, False,
        id="2d_i32_dim1_nokeepdim",
    ),
    pytest.param(
        (torch.randn(3, 4, 2, dtype=torch.float64)),
        "3x4x2xf64", "3x4xf64",
        2, False,
        id="3d_f64_dim2_nokeepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (4, 3, 2, 4), dtype=torch.int64)),
        "4x3x2x4xi64", "4x3x2xi64",
        3, False,
        id="4d_i64_dim3_nokeepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (5, 4, 3, 2, 4), dtype=torch.int64)),
        "5x4x3x2x4xi64", "5x3x2x4xi64",
        1, False,
        id="5d_i64_dim1_nokeepdim",
    ),
    # --- dim specified, keepdim=True: axis dim set to 1 ---
    pytest.param(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        "2x3xf32", "1x3xf32",
        0, True,
        id="2d_f32_dim0_keepdim",
    ),
    pytest.param(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        "2x3xi32", "2x1xi32",
        1, True,
        id="2d_i32_dim1_keepdim",
    ),
    pytest.param(
        (torch.randn(2, 3, 4, dtype=torch.float64)),
        "2x3x4xf64", "2x3x1xf64",
        2, True,
        id="3d_f64_dim2_keepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (4, 3, 2, 4), dtype=torch.int64)),
        "4x3x2x4xi64", "4x3x2x1xi64",
        3, True,
        id="4d_i64_dim3_keepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (5, 4, 3, 2, 4), dtype=torch.int64)),
        "5x4x3x2x4xi64", "5x1x3x2x4xi64",
        1, True,
        id="5d_i64_dim1_keepdim",
    ),
    # --- negative dim ---
    pytest.param(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "2x2xf32", "2xf32",
        -1, False,
        id="2d_f32_neg_dim1_nokeepdim",
    ),
    pytest.param(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "2x2xf32", "1x2xf32",
        -2, True,
        id="2d_f32_neg_dim0_keepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (4, 3, 2, 4), dtype=torch.int64)),
        "4x3x2x4xi64", "4x2x4xi64",
        -3, False,
        id="4d_i64_neg_dim1_nokeepdim",
    ),
    pytest.param(
        (torch.randint(1, 10, (5, 4, 3, 2, 4), dtype=torch.int64)),
        "5x4x3x2x4xi64", "5x4x3x2xi64",
        -1, False,
        id="5d_i64_neg_dim4_nokeepdim",
    ),
]


@pytest.mark.parametrize("input_tensor, in_type, out_type, dim, keepdim", PROD_TEST_CASES)
def test_prod(input_tensor, in_type, out_type, dim, keepdim):
    _, elem_type = parse_mlir_type(in_type)
    print("-" * 80)
    actual = run_prod(input_tensor, in_type, out_type, dim, keepdim)

    if dim is None:
        expected_scalar = torch.prod(input_tensor).item()
        expected = np.array(expected_scalar, dtype=Num_Dty[elem_type])
    else:
        expected = torch.prod(input_tensor, dim=dim, keepdim=keepdim).numpy()

    print(f"\n calc output  : {actual}")
    print(f" torch output : {expected}")
    print("-" * 80)

    assert actual is not None, "run_prod returned None — lowering failed"

    if actual.dtype in (np.float32, np.float64):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)
