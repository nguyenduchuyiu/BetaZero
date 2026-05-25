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
lemma numbertheory_4x3m7y3neq2003_1
  (x y : ℤ) :
  ¬(4 : ℤ) * x ^ (3 : ℕ) - (7 : ℤ) * y ^ (3 : ℕ) = (2003 : ℤ) := by
  intro h
  have h1 : (4 : ℤ) * x ^ 3 % 7 = 1 % 7 := by 
    have h2 : (7 : ℤ) * y ^ 3 % 7 = 0 := by 
      simp [Int.mul_emod, pow_three, Int.add_emod, Int.sub_emod]
    have h3 : (4 : ℤ) * x ^ 3 % 7 = (2003 : ℤ) % 7 := by 
      omega
    norm_num at h3 
    omega
  have h2 : (x ^ 3 : ℤ) % 7 = 0 ∨ (x ^ 3 : ℤ) % 7 = 1 ∨ (x ^ 3 : ℤ) % 7 = 6 := by 
    have h4 : (x % 7 = 0) ∨ (x % 7 = 1) ∨ (x % 7 = 2) ∨ (x % 7 = 3) ∨ (x % 7 = 4) ∨ (x % 7 = 5) ∨ (x % 7 = 6) := by 
      omega
    rcases h4 with (h4 | h4 | h4 | h4 | h4 | h4 | h4) <;> ( 
      norm_num [h4, pow_three, Int.add_emod, Int.mul_emod, Int.sub_emod] 
    )
  have h3 : (4 * (x ^ 3 : ℤ) % 7 = 1 % 7) := by 
    omega
  have h4 : (x ^ 3 : ℤ) % 7 = 2 := by 
    omega
  omega