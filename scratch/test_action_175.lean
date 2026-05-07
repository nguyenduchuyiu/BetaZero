import Mathlib
open Real

theorem test (a b : ℝ) 
  (h_subst0 : logb 8 a + 3 * logb 8 b = 5) 
  (h_log_a : logb 8 a = 2) : logb 8 b = 1 := by
  linarith
