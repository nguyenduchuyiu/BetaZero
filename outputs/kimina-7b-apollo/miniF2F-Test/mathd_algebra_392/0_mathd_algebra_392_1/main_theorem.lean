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
lemma mathd_algebra_392_1
  (n : ℕ)
  (h₀ : Even n)
  (h₁ : ((↑n : ℤ) - (2 : ℤ)) ^ (2 : ℕ) + (↑n : ℤ) ^ (2 : ℕ) + ((↑n : ℤ) + (2 : ℤ)) ^ (2 : ℕ) = (12296 : ℤ)) :
  (n - (2 : ℕ)) * n * (n + (2 : ℕ)) / (8 : ℕ) = (32736 : ℕ) := by
    have h2 : (n : ℤ) ^ 2 = 4096 := by
      have h_eq : ((↑n : ℤ) - 2) ^ 2 + (↑n : ℤ) ^ 2 + ((↑n : ℤ) + 2) ^ 2 = 12296 := by
        exact_mod_cast h₁
      ring_nf at h_eq
      nlinarith
    have hn : n = 64 := by
      have h_nsq : (n : ℤ) ^ 2 = 4096 := h2
      have h_n : (n : ℤ) ≥ 0 := by
        exact_mod_cast Nat.cast_nonneg n
      have h : (n : ℤ) = 64 := by
        nlinarith [sq_nonneg ((n : ℤ) - 64)]
      exact_mod_cast h
    calc
      (n - 2) * n * (n + 2) / 8
          = (64 - 2) * 64 * (64 + 2) / 8 := by rw [hn]
      _ = 62 * 64 * 66 / 8 := by norm_num
      _ = 32736 := by norm_num