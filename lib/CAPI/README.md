# Calc Dialect C API Implementation

This directory contains the C++ implementations of the C API bindings for the `calc` dialect. These bindings expose the internal C++ MLIR structures to a stable C interface, which is primarily used to generate and interact with the Python bindings.

## File Breakdown

* **`Dialects.cpp`**: Implements the C API for registering and interacting with the `calc` dialect itself.
* **`Registration.cpp`**: Implements the C API for registering the dialect with an MLIR context.
* **`Transforms.cpp`**: Implements the C API for registering and applying the custom optimization and lowering passes.
