import Mathlib

open BigOperators Nat Real Topology

theorem my_theorem (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 < w) (h0 : logb ↑x ↑w = 24) (h1 : logb ↑y ↑w = 40) (h2 : logb (↑x * ↑y * ↑z) ↑w = 12) (h_base_x : logb ↑w ↑x = 1 / 24) (hy_pos : 0 < ↑y) (hy_ne_one : ↑y ≠ 1) (hw_pos : 0 < ↑w) : ↑w ≠ 1 := by
  intro h_w_one
  have h_w_one_f : (w : ℝ) = 1 := by norm_cast
  rw [h_w_one_f, logb_one] at h0
  norm_num at h0
