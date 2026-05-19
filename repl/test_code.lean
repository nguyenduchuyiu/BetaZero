import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Nat Real Topology

theorem test_action_174_skeleton
  (f : ℝ → ℝ)
  (h₀ : ∀ (x : ℝ), f x = x ^ 2 + (18 * x + 30) - 2 * √(x ^ 2 + (18 * x + 45)))
  (h₁ : Fintype ↑(f ⁻¹' {0}))
  (x1 x2 : ℝ)
  (h_equiv : ∀ (x : ℝ), f x = 0 ↔ x ^ 2 + 18 * x + 20 = 0)
  (h_solve : ∀ (x : ℝ), x ^ 2 + 18 * x + 20 = 0 ↔ x = -9 + √61 ∨ x = -9 - √61)
  : ({x1, x2} : Set ℝ) = ({-9 + √61, -9 - √61} : Set ℝ) := by
  have h_subset1 : {x1, x2} ⊆ ({-9 + √61, -9 - √61} : Set ℝ) := by admit
  have h_subset2 : ({-9 + √61, -9 - √61} : Set ℝ) ⊆ {x1, x2} := by admit
  exact Set.Subset.antisymm h_subset1 h_subset2

theorem test_action_174_with_action_255
  (f : ℝ → ℝ)
  (h₀ : ∀ (x : ℝ), f x = x ^ 2 + (18 * x + 30) - 2 * √(x ^ 2 + (18 * x + 45)))
  (h₁ : Fintype ↑(f ⁻¹' {0}))
  (x1 x2 : ℝ)
  (h_equiv : ∀ (x : ℝ), f x = 0 ↔ x ^ 2 + 18 * x + 20 = 0)
  (h_solve : ∀ (x : ℝ), x ^ 2 + 18 * x + 20 = 0 ↔ x = -9 + √61 ∨ x = -9 - √61)
  : ({x1, x2} : Set ℝ) = ({-9 + √61, -9 - √61} : Set ℝ) := by
  have h_subset1 : {x1, x2} ⊆ ({-9 + √61, -9 - √61} : Set ℝ) := by admit
  have h_subset2 : ({-9 + √61, -9 - √61} : Set ℝ) ⊆ {x1, x2} := by
    intro x hx
    have h_card_root : ({-9 + √61, -9 - √61} : Set ℝ).ncard = 2 := by
      apply Set.ncard_pair
      intro h_eq
      have h_zero : √61 = 0 := by linarith
      have h_61 : 61 = 0 := (sqrt_eq_zero (by linarith)).mp h_zero
      norm_num at h_61
    have h_card_x12 : ({x1, x2} : Set ℝ).ncard ≤ 2 := Set.ncard_insert_le x1 {x2}
    have h_eq_sets : ({x1, x2} : Set ℝ) = {-9 + √61, -9 - √61} := by
      apply Set.eq_of_subset_of_ncard_le h_subset1
      rw [h_card_root]
      exact h_card_x12
    rw [h_eq_sets]
    exact hx
  exact Set.Subset.antisymm h_subset1 h_subset2
