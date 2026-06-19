# Calc Dialect Test Suite

This directory contains the comprehensive testing infrastructure for the `calc` MLIR dialect. 

The project utilizes LLVM's `lit` (LLVM Integrated Tester) paired with `FileCheck` to verify the C++ compiler transformations, and `pytest` (or a similar runner) to validate the Python API bindings.

## Directory Structure

* **`calc/`**: Contains the core MLIR test files (`.mlir`). These are "lit tests" that invoke the `calc-opt` command-line tool to run specific passes, and then pipe the output to `FileCheck` to assert that the resulting IR matches the expected structure. These tests typically validate:
  * Correct parsing and printing of custom operations.
  * Custom operation verifiers (e.g., failing compilation on invalid shapes).
  * The precise rewrite patterns of lowering passes like `--calc-to-tosa` or `--calc-broadcast`.
* **`python/`**: Contains the Python test suite (`.py` files). These tests verify that the `nanobind` Python wrappers correctly expose the dialect, ensuring that `calc` operations can be seamlessly constructed, manipulated, and compiled programmatically from a Python environment.
* **`lit.cfg.py` & `lit.site.cfg.py.in`**: Configuration files that initialize the `lit` testing environment, setting up required paths to binaries like `calc-opt` and `FileCheck`.
