#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "calc-c/Dialects.h"
#include "calc-c/Registration.h"

namespace nb = nanobind;

// This file defines the Python bindings for the calc-mlir module, which includes the registration of the calc dialect 
// and its associated passes. 
NB_MODULE(_calcMlir, m) {
  calcRegisterAllPasses();

  m.doc() = "calc-mlir python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__calc__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
}