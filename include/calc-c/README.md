# Calc Dialect C API

This directory contains the C API headers for the `calc` MLIR dialect. 

These headers export a stable, C-compatible interface for the dialect's core functionality. This C layer is essential for creating language bindings (most notably Python bindings), allowing the `calc` dialect's operations, passes, and types to be constructed and manipulated outside of a pure C++ environment.

## File Breakdown

* **`Dialects.h`**: Provides the C API for registering and interacting with the `calc` dialect itself.
* **`Registration.h`**: Provides C API entry points for registering the dialect with an opaque MLIR context.
* **`Transforms.h`**: Provides the C API for registering and applying the custom optimization and lowering passes defined within the `calc` dialect.

## Usage

These headers are typically consumed by the corresponding C API implementation (usually located in a `lib/CAPI/` directory) and the Python bindings generation layer. They rely on the opaque types (like `MlirContext`, `MlirDialect`, and `MlirPass`) provided by the upstream core MLIR C API.
