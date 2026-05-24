import Mathlib
import Aesop

open BigOperators Nat Real Topology

theorem my_theorem (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 < w) (h0 : Real.logb x w = 24) (h1 : Real.logb y w = 40) (h2 : Real.logb (x * y * z) w = 12) : Real.logb z w = 60 := by
  have h_base_x : logb ↑w ↑x = 1 / 24 := by
    have h_base_x : 0 < (x : ℝ) ∧ (x : ℝ) ≠ 1 := sorry
    have h_val_w : 0 < (w : ℝ) ∧ (w : ℝ) ≠ 1 := by
      constructor
      · norm_cast
      · intro hw1
        have h_log_w : logb ↑x ↑w = logb ↑x 1 := by rw [hw1, Nat.cast_one]
        rw [h_log_w, logb_one] at h0
        norm_num at h0
    sorry
  sorry
