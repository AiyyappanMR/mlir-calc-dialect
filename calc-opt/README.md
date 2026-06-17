# calc-opt

This directory contains the source code for the `calc-opt` tool. 

`calc-opt` is a standalone command-line utility used for testing, analyzing, and transforming MLIR code that utilizes the `calc` dialect. It serves the exact same purpose for this project as `mlir-opt` does for upstream MLIR.

## Purpose

The primary responsibilities of `calc-opt` are:
1. **Dialect Registration**: It registers the custom `calc` dialect (alongside core MLIR dialects like `func`, `arith`, and `tensor`) so that the MLIR parser can understand and construct `.mlir` files containing `calc` operations.
2. **Pass Registration**: It registers all the optimization, lowering, and conversion passes specific to the `calc` dialect.
3. **Pipeline Execution**: It provides a CLI to load an MLIR module, run a user-specified sequence of passes over it, and output the transformed IR.

## Usage

`calc-opt` is heavily utilized by the `lit` test suite (in the `test/` directory) to verify the behavior of operations and passes via `FileCheck`.

A typical manual invocation looks like this:

```bash
# Run a specific calc pass on an input file and print the transformed IR
./bin/calc-opt --calc-some-pass input.mlir > output.mlir

# Run a sequence of passes
./bin/calc-opt --calc-pass-one --calc-pass-two input.mlir
```

To see a full list of all registered dialects, passes, and command-line options, run:

```bash
./bin/calc-opt --help
```
