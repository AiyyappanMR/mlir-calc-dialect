file(REMOVE_RECURSE
  "CMakeFiles/MLIRcalcOpsIncGen"
  "calcOps.cpp.inc"
  "calcOps.h.inc"
  "calcOpsDialect.cpp.inc"
  "calcOpsDialect.h.inc"
  "calcOpsTypes.cpp.inc"
  "calcOpsTypes.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRcalcOpsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
