# Calc Dialect Core Definitions

This directory contains the core TableGen (`.td`) and C++ header (`.h`) files that define the `calc` MLIR dialect. 

These files form the primary interface for creating, manipulating, and optimizing `calc` operations.

## File Breakdown

* **Dialect Definition**
  * `calcDialect.td`: TableGen definition for the `calc` dialect itself.
  * `calcDialect.h`: C++ header declaring the dialect class and registering it with MLIR.

* **Operations**
  * `calcOps.td`: TableGen definitions for all operations within the `calc` dialect (e.g., mathematical operations, types, attributes).
  * `calcOps.h`: C++ header declaring the operation classes generated from the TableGen definitions.

* **Transformations & Passes**
  * `calcPasses.td`: TableGen definitions for optimization and lowering passes specific to the `calc` dialect.
  * `calcPasses.h`: C++ header providing declarations and registration entry points for these passes.
  * `calcPatterns.td`: Declarative Rewrite Rules (DRR) in TableGen used to define pattern-matching transformations for `calc` operations.

## Usage

When developing tools or linking against the `calc` dialect in C++, you will typically include the relevant headers from this directory:

```cpp
#include "calc/calcDialect.h"
#include "calc/calcOps.h"
#include "calc/calcPasses.h"
```

## Available Operations

| Operation | Description |
|---|---|
| `calc.const` | To get literal value |
| `calc.add` | Add two scalar values |
| `calc.addt` | Add two tensor values |
| `calc.mult` | Multiply two tensor values |
| `calc.addcmul` | Add scaled element-wise multiplication of two tensors to input |
| `calc.logaddexp2` | Add two tensors in log space with base 2 |
| `calc.minimum` | Performs element-wise minimum of two tensors |
| `calc.entr` | Computes the element-wise entropy of a tensor |
| `calc.split` | Splits a tensor into chunks of specified sizes along a dimension |
| `calc.catmuladd` | Concatenates two variadic tensor groups separately, multiplies the results element-wise, and optionally adds a scale tensor to the product |
| `calc.prod` | Computes the product of all elements in the input tensor |
| `calc.softmax` | Computes the softmax of the input tensor along the specified dimension |
| `calc.stack` | Stacks a list of tensors along a new dimension |
| `calc.mul` | Multiply two values |
| `calc.print` | Prints the result |

## Available Passes

| Pass | Description |
|---|---|
| `--calc-to-arith` | Lower calc dialect to arith dialect |
| `--calc-to-tosa` | Lower calc dialect to tosa dialect |
| `--calc-broadcast` | Broadcast calc operations to have matching shapes |
| `--calc-verify-tosa-backend-contract` | Verifies that all calc ops have been lowered to TOSA. |
| `--calc-bufferize` | Bufferize calc operations |
| `--calc-to-memref` | Lower bufferized calc ops to memref ops |

*Note: The actual source code implementing the logic declared in these headers is located in the project's root `lib/` directory.*


