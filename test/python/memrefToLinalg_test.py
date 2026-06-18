# This file contains pytest tests for functional testing of the memref-to-linalg
# conversion pass, which rewrites memref.copy ops into linalg.generic ops with
# identity indexing maps and parallel iterator types.

import os
import sys
import ctypes
import pytest
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
#                 Pipeline: memref-to-linalg → LLVM
#---------------------------------------------------------------------------------

def MemRefToLinalgToLLVM(module):
    pm = PassManager.parse(
        "builtin.module("
            "memref-to-linalg,"
            "func.func(convert-linalg-to-loops),"
            "func.func(convert-scf-to-cf),"
            "expand-strided-metadata,"
            "finalize-memref-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-index-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


#---------------------------------------------------------------------------------
#                 IR Builder
#---------------------------------------------------------------------------------

def build_ir(memref_type):
    ir = f"""
func.func @test_memref_copy(%arg0: {memref_type}) -> {memref_type} attributes {{llvm.emit_c_interface}} {{
  %0 = memref.alloc() : {memref_type}
  memref.copy %arg0, %0 : {memref_type} to {memref_type}
  return %0 : {memref_type}
}}
"""
    return ir


#---------------------------------------------------------------------------------
#                 Runner
#---------------------------------------------------------------------------------

def run_memref_copy(input_array, memref_type):

    ir = build_ir(memref_type)
    result = np.zeros_like(input_array)

    with Context() as ctx:

        register_dialect(ctx)

        # parse and lower all the way to LLVM IR
        module = Module.parse(ir)
        execution_engine = ExecutionEngine(
            MemRefToLinalgToLLVM(module),
            shared_libs=["/home/mcw/llvm-project/build/lib/libmlir_c_runner_utils.so"],
        )

        # build memref descriptors for input and output
        mem_input  = get_ranked_memref_descriptor(input_array)
        mem_result = get_ranked_memref_descriptor(result)

        # result goes first — tensor return ABI convention for emit_c_interface
        execution_engine.invoke(
            "test_memref_copy",
            ctypes.pointer(ctypes.pointer(mem_result)),
            ctypes.pointer(ctypes.pointer(mem_input)),
        )

        return ranked_memref_to_numpy(ctypes.pointer(mem_result))


#---------------------------------------------------------------------------------
#                 Test Cases
#---------------------------------------------------------------------------------
MEMREF_COPY_TEST_CASES = [

    # 1D f32
    (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "memref<10xf32>",
    ),

    # 2D f32
    (
        np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
        "memref<3x4xf32>",
    ),

    # 3D i32
    (
        np.ones((2, 3, 4), dtype=np.int32),
        "memref<2x3x4xi32>",
    ),

    # 4D i64
    (
        np.arange(120, dtype=np.int64).reshape(2, 3, 4, 5),
        "memref<2x3x4x5xi64>",
    ),
]


#---------------------------------------------------------------------------------
#                 Pytest
#---------------------------------------------------------------------------------

@pytest.mark.parametrize("input_array, memref_type", MEMREF_COPY_TEST_CASES)
def test_memref_copy(input_array, memref_type):

    actual   = run_memref_copy(input_array, memref_type)
    expected = input_array  # copy should produce identical values

    print(f"\n calc output  : {actual}")
    print(f" expected     : {expected}")
    print("-" * 80)

    # verify lowering didn't fail
    assert actual is not None, "run_memref_copy returned None — lowering failed"

    # compare actual vs expected — use exact match for int, tolerance for float
    if np.issubdtype(input_array.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)