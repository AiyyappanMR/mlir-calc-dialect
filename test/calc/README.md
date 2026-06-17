# Calc Dialect LIT Tests

This directory contains the MLIR "lit" (LLVM Integrated Tester) tests for the `calc` dialect. 

These tests are written directly in MLIR syntax (`.mlir` files). They typically invoke the `calc-opt` command-line tool at the top of the file using `// RUN:` directives to execute specific optimization or lowering passes, and pipe the output to `FileCheck` to ensure the compiler's transformed IR matches the expected structure.

## Test Focus

The files in this directory are primarily focused on verifying **Transformation and Lowering Passes**. Rather than just testing basic parsing, they validate that the C++ rewrite patterns correctly convert `calc` operations into their lower-level equivalents (like `tosa` or `arith`).

### 1. Operation-Specific Lowering Tests
These files isolate specific complex `calc` operations and verify that they lower correctly into the `tosa` dialect across various edge cases (such as different reduction dimensions, varying tensor ranks, negative indexing, or different element types).
* **Examples**: `split.mlir`, `stack.mlir`, `softmax.mlir`, `catmuladd.mlir`, `prod.mlir`, `entr.mlir`, `minimum.mlir`, `logaddexp2.mlir`. 

### 2. General Pass Tests
These files test specific cross-operation passes or broader lowering pipelines.
* **Examples**: 
  * `test.mlir`, `CalcToArith_test.mlir`: Tests lowering scalar `calc` ops down to the standard `arith` dialect.
  * `broadcast.mlir`: Tests the `--calc-broadcast` pass, ensuring explicit reshape or broadcast operations are correctly inserted for mismatched operand shapes.
  * `canonicalizer_test.mlir`: Tests constant folding and algebraic simplification patterns.
  * `verifyTosaBackendContract.mlir`: Validates the strict verification pass (`--calc-verify-tosa-backend-contract`) ensuring it correctly errors out if unlowered `calc` ops are found.
