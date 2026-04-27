import sys
import os

# point to the generated python bindings in the build dir
CALC_PYTHON_PATH = "/home/mcw/Desktop/mlir-calc-dialect/build/python"
if CALC_PYTHON_PATH not in sys.path:
    sys.path.insert(0, CALC_PYTHON_PATH)

# point to the core mlir python packages
MLIR_CORE_PATH = "/home/mcw/llvm-project/build/tools/mlir/python_packages/mlir_core"
if MLIR_CORE_PATH not in sys.path:
    sys.path.insert(0, MLIR_CORE_PATH)

from mlir.ir import Context, Module, Location
from mlir.passmanager import PassManager
from mlir.dialects._ods_common import _cext

# register our custom dialect with the mlir 
_cext.globals.append_dialect_search_prefix("calc_dialect.dialects")

# load our generated calc dialect wrappers
from calc_dialect.dialects import calc

def test_ods_bindings():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True

        # test that mlir can parse raw strings into our dialect ops
        ir = """
        module {
          func.func @test_add(%arg0: f32, %arg1: f32) -> f32 {
            %0 = "calc.add"(%arg0, %arg1) : (f32, f32) -> f32
            return %0 : f32
          }
        }
        """
        module = Module.parse(ir)
        # lower our custom calc dialect ops to standard arith ops using our registered pass
        pm = PassManager.parse("builtin.module(calc-to-arith)")
        pm.run(module.operation)
        print(module)

if __name__ == "__main__":
    test_ods_bindings()
