# RUN: %PYTHON %s | calc-opt -split-input-file | FileCheck

import gc
import sys
import ctypes
import subprocess    
import torch 
import numpy as np                 

MLIR_CORE_PATH = "/home/mcw/llvm-project/build/tools/mlir/python_packages/mlir_core"
if MLIR_CORE_PATH not in sys.path:
    sys.path.insert(0, MLIR_CORE_PATH)

from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy

def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

def run(f):
  log("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0

def lowerToLLVM(module):
  pm = PassManager.parse(
      "builtin.module(convert-complex-to-llvm,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)")
  pm.run(module.operation)
  return module

# CHECK-LABEL: TEST: test_FloatAdd
def test_FloatAdd():
  with Context():
    module = Module.parse(r"""
func.func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : f32
  return %add : f32
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(42.)
    arg1 = c_float_p(2.)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2.0 = 43.0
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))


run(test_FloatAdd)

