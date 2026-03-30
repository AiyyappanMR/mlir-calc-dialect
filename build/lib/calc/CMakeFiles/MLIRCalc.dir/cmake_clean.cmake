file(REMOVE_RECURSE
  "../libMLIRCalc.a"
  "../libMLIRCalc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRCalc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
