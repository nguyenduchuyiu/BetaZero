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
lemma amc12a_2020_p4_1
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ (1000 : ℕ) ≤ n ∧ n ≤ (9999 : ℕ) ∧ (∀ d ∈ digits (10 : ℕ) n, Even d) ∧ (5 : ℕ) ∣ n) :
  S.card = (100 : ℕ) := by
  have h1 : S = Finset.filter (fun n => (1000 : ℕ) ≤ n ∧ n ≤ (9999 : ℕ) ∧ (∀ d ∈ digits (10 : ℕ) n, Even d) ∧ (5 : ℕ) ∣ n) (Finset.Icc 1000 9999) := by
    ext n
    simp [h₀]
    omega
  rw [h1]
  rw [Finset.filter]
  simp [Finset.Icc]
  native_decide