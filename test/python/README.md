# Calc Dialect Python Tests

This directory contains the Python-based functional test suite for the `calc` MLIR dialect.

Unlike the lit tests in `test/calc/` (which only verify the IR structure after lowering), these tests verify the **numerical correctness** of the `calc` operations. They compile the operations all the way down to machine code and compare their execution outputs against reference implementations in PyTorch.

## Test Architecture

The test files in this directory generally follow this execution flow:

1. **IR Construction**: They generate raw MLIR string templates containing the `calc` operations (e.g., `"calc.split"(%arg0)`).
2. **Parsing & Lowering**: The MLIR string is parsed into an `mlir.ir.Module`. A comprehensive `PassManager` pipeline is then invoked from Python to lower the `calc` operations through TOSA, Linalg, SCF, Arith, and finally down to LLVM IR.
3. **JIT Execution**: The lowered module is passed to the MLIR `ExecutionEngine` which JIT-compiles it. The engine is invoked, passing in Numpy arrays as inputs.
4. **Validation**: The numerical output returned by the `ExecutionEngine` is compared against the expected output produced by the equivalent PyTorch operator (e.g., comparing `calc.split` against `torch.split`) using `pytest`.

## Test Categories

* **Operation Functional Tests**: Files like `calc_split_test.py`, `calc_stack_test.py`, `calc_catmuladd_test.py`, `calc_softmax_test.py`, `calc_prod_test.py`, etc., thoroughly test a specific operation across varying shapes, ranks, and data types (f32, f64, i32, i64).

## Execution

These tests are standard Python parameterized tests executed using **`pytest`**. 

You can run them directly via:
```bash
pytest test/python/
```
