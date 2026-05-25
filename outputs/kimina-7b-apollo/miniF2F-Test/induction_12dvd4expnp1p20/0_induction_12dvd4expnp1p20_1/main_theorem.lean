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
lemma induction_12dvd4expnp1p20_1
  (n : ℕ) :
  (12 : ℕ) ∣ (4 : ℕ) ^ (n + (1 : ℕ)) + (8 : ℕ) := by
    induction n with
    | zero =>
      norm_num
    | succ n ih =>
      have h1 : (4 : ℕ) ^ (n + 1 + 1) + 8 = 4 * (4 ^ (n + 1)) + 8 := by
        ring
      rw [h1]
      omega