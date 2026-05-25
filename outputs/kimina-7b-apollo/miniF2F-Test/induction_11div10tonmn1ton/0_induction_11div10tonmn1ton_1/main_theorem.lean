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
lemma induction_11div10tonmn1ton_1
  (n : ℕ) :
  (11 : ℤ) ∣ -(-1 : ℤ) ^ n + (10 : ℤ) ^ n := by
    induction n with
    | zero =>
      use 0
      simp
    | succ k ih =>
      have h1 : -(-1 : ℤ) ^ (k + 1) + (10 : ℤ) ^ (k + 1) = -(-1 : ℤ) ^ k * (-1) + (10 : ℤ) * (10 : ℤ) ^ k := by
        ring_nf
      rw [h1]
      omega