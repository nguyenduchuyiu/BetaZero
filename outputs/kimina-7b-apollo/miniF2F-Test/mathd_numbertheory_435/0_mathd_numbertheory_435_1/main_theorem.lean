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
lemma mathd_numbertheory_435_1
  (k : ℕ)
  (h₀ : (0 : ℕ) < k)
  (h₁ : ∀ (n : ℕ), ((6 : ℕ) * n + k).gcd ((6 : ℕ) * n + (3 : ℕ)) = (1 : ℕ))
  (h₂ : ∀ (n : ℕ), ((6 : ℕ) * n + k).gcd ((6 : ℕ) * n + (2 : ℕ)) = (1 : ℕ))
  (h₃ : ∀ (n : ℕ), ((6 : ℕ) * n + k).gcd ((6 : ℕ) * n + (1 : ℕ)) = (1 : ℕ)) :
  (5 : ℕ) ≤ k := by 
  by_contra h
  push_neg at h
  interval_cases k <;> 
  try { 
    specialize h₁ 0 
    simp at h₁ 
    all_goals 
      contradiction 
  }
  <;> try { 
    specialize h₂ 1 
    simp at h₂ 
    all_goals 
      contradiction 
  }
  <;> try { 
    specialize h₃ 1 
    simp at h₃ 
    all_goals 
      contradiction 
  }