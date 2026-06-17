# mlir-calc-dialect Includes

This directory contains the public C++ and C API headers, as well as the TableGen (`.td`) definitions for the `calc` MLIR dialect.

## Directory Structure

* **`calc/`**: Contains the core C++ headers and TableGen files that define the `calc` dialect.
  * `calcDialect.td` / `calcDialect.h`: Dialect registration and definition.
  * `calcOps.td` / `calcOps.h`: Operation definitions and traits.
  * `calcPasses.td` / `calcPasses.h`: Optimization and transformation passes.
  * `calcPatterns.td`: Declarative rewrite patterns (DRR).
* **`calc-c/`**: Contains C API headers for the dialect, which are useful for creating C bindings (e.g., for Python integration).
  * `Dialects.h`
  * `Registration.h`
  * `Transforms.h`

These headers define the public interface of the dialect and are meant to be included by the implementation files (in `lib/`) as well as external tools and downstream projects utilizing the `calc` dialect.
