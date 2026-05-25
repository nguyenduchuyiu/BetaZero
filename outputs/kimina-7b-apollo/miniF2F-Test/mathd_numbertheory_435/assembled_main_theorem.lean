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
theorem mathd_numbertheory_435
  (k : ℕ)
  (h₀ : 0 < k)
  (h₁ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 3) = 1)
  (h₂ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 2) = 1)
  (h₃ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 1) = 1) :
  5 ≤ k := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
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

