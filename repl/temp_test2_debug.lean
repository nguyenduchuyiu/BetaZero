import Mathlib
import Aesop

open BigOperators Nat Real Topology

theorem my_theorem (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 < w) (h0 : Real.logb x w = 24) (h1 : Real.logb y w = 40) (h2 : Real.logb (x * y * z) w = 12) (h_base_x : 0 < ↑x ∧ ↑x ≠ 1) : 0 < ↑w ∧ ↑w ≠ 1 := by
  constructor
  · norm_cast
  · intro hw1
    have h_log_w : logb ↑x ↑w = logb ↑x 1 := by
      #check hw1
      #check (1 : ℝ)
      #check (1 : ℕ)
      #check logb ↑x ↑w
      #check logb ↑x 1
      rw [hw1]
      -- Let's see if this compiles or what the state is here
      sorry
    sorry
