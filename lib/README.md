# mlir-calc-dialect Library Implementation

This directory contains the core source code (`.cpp` files) that implements the logic and behavior defined by the headers in the `include/` directory. 

While the `include/` directory provides the declarations, interfaces, and TableGen structural definitions, the `lib/` directory provides the actual C++ logic for the dialect's operations, verification rules, transformations, and bindings.

## Directory Structure

* **`calc/`**: Contains the core C++ implementations for the `calc` dialect and its passes.
  * `calcDialect.cpp`: Implements the dialect registration and initialization logic.
  * `calcOps.cpp`: Implements the custom behavior, verifiers, folders, and canonicalizers for the operations defined in `calcOps.td`.
  * `broadcast.cpp`: Implements the `--calc-broadcast` pass logic to ensure matching operand shapes before lowering.
  * `calcToArith.cpp`: Implements the lowering patterns from `calc` ops to the `arith` dialect (`--calc-to-arith`).
  * `calcToTosa.cpp`: Implements the lowering patterns from `calc` ops to the `tosa` dialect (`--calc-to-tosa`).
  * `verifyTosaBackendContract.cpp`: Implements the strict verification pass (`--calc-verify-tosa-backend-contract`) that ensures no `calc` operations remain in the IR after lowering to the TOSA backend.
  * `calcBufferize.cpp`: Implements the `--calc-bufferize` pass to bufferize tensor-based calc operations.
  * `calcToMemRef.cpp`: Implements the `--calc-to-memref` pass to lower bufferized calc ops to memref operations.

* **`CAPI/`**: Contains the implementations for the C API bindings declared in `include/calc-c/`. 
  * `Dialects.cpp`: Implements the C API for dialect interaction.
  * `Registration.cpp`: Implements the C API for registering the dialect with an MLIR context.
  * `Transforms.cpp`: Implements the C API for pass registration and pipeline execution.
  
## Usage

The code in this directory is compiled into static or shared libraries (e.g., `MLIRCalc`, `MLIRCalcCAPI`). These libraries are then linked into tools like `calc-opt`, integrated into downstream projects, or loaded dynamically by the Python bindings.
