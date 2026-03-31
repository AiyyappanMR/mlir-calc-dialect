// RUN: ./bin/calc-opt ../test/affine.mlir
// RUN(lower affine -> arith,scf -> arith,cf): 
// /bin/calc-opt ../test/affine.mlir   --pass-pipeline="builtin.module(func.func(
//     lower-affine,convert-scf-to-cf),
//     convert-cf-to-llvm, convert-arith-to-llvm, convert-func-to-llvm
//   )" --mlir-print-ir-after-all


#map1 = affine_map<(d0) -> (d0+1)>

module {
  func.func @main() {
    %A = memref.alloc() : memref<10xf32>

    affine.for %i = 0 to 10 {
      // CHECK: %{{.*}} = arith.constant 1 : index
      %j = affine.apply #map1(%i)
      %a = affine.load %A[%j] : memref<10xf32>
    }

    return
  }
}