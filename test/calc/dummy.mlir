// This should PASS because we are passing floats to a float-only Op
func.func @test_ok(%arg0: f32, %arg1: f32) {
  %0 = "calc.add"(%arg0, %arg1) : (f32, f32) -> f32
  return
}

// This should FAIL because we are passing integers to a float-only Op
func.func @test_error(%arg0: i32, %arg1: i32) {
  %0 = "calc.add"(%arg0, %arg1) : (i32, i32) -> i32
  return
}