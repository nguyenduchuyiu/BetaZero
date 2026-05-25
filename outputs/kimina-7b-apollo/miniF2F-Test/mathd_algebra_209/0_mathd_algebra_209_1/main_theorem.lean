import Mathlib
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
set_option pp.instanceTypes true
set_option pp.numericTypes true
set_option pp.coercions.types true
set_option pp.letVarTypes true
set_option pp.structureInstanceTypes true
set_option pp.instanceTypes true
set_option pp.mvars.withType true
set_option pp.coercions true
set_option pp.funBinderTypes true
set_option pp.piBinderTypes true
lemma mathd_algebra_209_1
  (σ : ℝ ≃ ℝ)
  (h₀ : (σ.symm : ℝ → ℝ) (2 : ℝ) = (10 : ℝ))
  (h₁ : (σ.symm : ℝ → ℝ) (10 : ℝ) = (1 : ℝ))
  (h₂ : (σ.symm : ℝ → ℝ) (1 : ℝ) = (2 : ℝ)) :
  (σ : ℝ → ℝ) ((σ : ℝ → ℝ) (10 : ℝ)) = (1 : ℝ) := by
    have h3 : (σ : ℝ → ℝ) (10 : ℝ) = (2 : ℝ) := by
      have h₃ : (σ.symm) 2 = (10 : ℝ) := by
        simpa using h₀
      have h₄ : (σ 10) = (2 : ℝ) := by
        have h₅ : σ 10 = 2 := by
          rw [←h₃]
          apply Equiv.right_inv _ 2
        exact h₅
      exact h₄
    have h4 : (σ : ℝ → ℝ) (2 : ℝ) = (1 : ℝ) := by
      have h₆ : (σ.symm) 1 = (2 : ℝ) := by
        simpa using h₂
      have h₇ : (σ 2) = (1 : ℝ) := by
        have h₈ : σ 2 = 1 := by
          rw [←h₆]
          apply Equiv.right_inv _ 1
        exact h₈
      exact h₇
    rw [h3]
    exact h4