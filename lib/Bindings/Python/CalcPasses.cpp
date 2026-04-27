#include <nanobind/nanobind.h>
#include "calc-c/Passes.h"

// This is responsible for exposing the C-API functions to Python using nanobind.
NB_MODULE(_calcPasses, m) {
    m.def("register_calc_passes", []() {
        mlirRegisterCalcPasses(); // Call the C-API function
    });
}