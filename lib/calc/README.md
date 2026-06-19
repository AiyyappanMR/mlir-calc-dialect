# Calc Dialect Implementation

This directory contains the C++ implementation source files (`.cpp`) for the `calc` MLIR dialect. 

These files provide the concrete logic that backs the operations, verifiers, folders, and transformations declared in the `include/calc/` headers and TableGen definitions.

## File Breakdown

### Core Dialect and Operations
* **`calcDialect.cpp`**: Implements the initialization and registration of the `calc` dialect within an `mlir::MLIRContext`.
* **`calcOps.cpp`**: Provides the C++ implementation for the custom operations. This includes complex custom verifiers (e.g., verifying shapes and dimension attributes in ops like `calc.split`, `calc.stack`, or `calc.softmax`), custom builder methods, and constant folding/canonicalization logic.

### Lowering and Transformation Passes
* **`broadcast.cpp`**: Implements the `--calc-broadcast` pass. This pass is responsible for inspecting `calc` operations that support implicit broadcasting and explicitly inserting reshape or expand operations in the IR to ensure operand shapes match exactly before further lowering.
* **`calcToArith.cpp`**: Implements the `--calc-to-arith` pass, providing the rewrite patterns to lower appropriate `calc` operations (such as simple scalar `calc.add` or `calc.mul`) down to the standard MLIR `arith` dialect.
* **`calcToTosa.cpp`**: Implements the `--calc-to-tosa` pass. This is the primary lowering trajectory for the dialect, converting complex tensor-based `calc` operations into their equivalents in the TOSA (Tensor Operator Set Architecture) dialect.
* **`verifyTosaBackendContract.cpp`**: Implements the `--calc-verify-tosa-backend-contract` pass. This is a strict verification step that runs after lowering to ensure that absolutely no `calc` operations remain in the IR, guaranteeing that the module is fully compliant with the expected TOSA backend contract.
* **`calcBufferize.cpp`**: Implements the `--calc-bufferize` pass to bufferize tensor-based calc operations.
* **`calcToMemRef.cpp`**: Implements the `--calc-to-memref` pass to lower bufferized calc ops to memref operations.

## Compilation

The source files in this directory are compiled together into the `MLIRCalc` library (as configured in the local `CMakeLists.txt`). This library is then statically linked into the `calc-opt` tool and the CAPI wrapper libraries.
