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
lemma algebra_2varlineareq_fp3zeq11_3tfm1m5zeqn68_feqn10_zeq7_1
  (f z : ℂ)
  (h₀ : f + (3 : ℂ) * z = (11 : ℂ))
  (h₁ : (3 : ℂ) * (f - (1 : ℂ)) - (5 : ℂ) * z = (-68 : ℂ)) :
  f = (-10 : ℂ) ∧ z = (7 : ℂ) := by
    have h2 : f = 11 - 3 * z := by 
      rw [←h₀]
      ring
    have hz : z = 7 := by 
      rw [h2] at h₁
      simp [Complex.ext_iff] at h₁ ⊢ 
      constructor
      · linarith [h₁]
      · linarith [h₁]
    have hf : f = -10 := by 
      rw [hz] at h2
      rw [h2]
      ring
    exact ⟨hf, hz⟩