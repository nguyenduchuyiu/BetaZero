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
lemma algebra_absapbon1pabsapbleqsumabsaon1pabsa_1
  (a b : ℝ) :
  |a + b| / ((1 : ℝ) + |a + b|) ≤ |a| / ((1 : ℝ) + |a|) + |b| / ((1 : ℝ) + |b|) := by
  have h1 : |a + b| ≤ |a| + |b| := abs_add a b
  have h2 : |a + b| / ((1 : ℝ) + |a + b|) ≤ (|a| + |b|) / ((1 : ℝ) + |a| + |b|) := by
    have h3 : |a + b| ≤ |a| + |b| := abs_add a b
    have h4 : (1 : ℝ) + |a + b| ≤ (1 : ℝ) + |a| + |b| := by linarith [abs_add a b]
    have h5 : 0 ≤ |a + b| := abs_nonneg (a + b)
    have h6 : 0 ≤ (1 : ℝ) + |a + b| := by linarith
    have h7 : 0 ≤ (1 : ℝ) + |a| + |b| := by
      have h8 : 0 ≤ |a| := abs_nonneg a
      have h9 : 0 ≤ |b| := abs_nonneg b
      linarith
    apply (div_le_div_iff (by linarith) (by linarith)).mpr
    nlinarith [abs_nonneg (a + b), abs_nonneg a, abs_nonneg b]
  have h3 : (|a| + |b|) / ((1 : ℝ) + |a| + |b|) ≤ |a| / ((1 : ℝ) + |a|) + |b| / ((1 : ℝ) + |b|) := by
    have h4 : 0 ≤ |a| := abs_nonneg a
    have h5 : 0 ≤ |b| := abs_nonneg b
    have h6 : 0 ≤ (1 : ℝ) + |a| := by
      linarith [abs_nonneg a]
    have h7 : 0 ≤ (1 : ℝ) + |b| := by
      linarith [abs_nonneg b]
    have h8 : (|a| + |b|) / ((1 : ℝ) + |a| + |b|) = |a| / ((1 : ℝ) + |a| + |b|) + |b| / ((1 : ℝ) + |a| + |b|) := by
      field_simp
      <;> linarith
    rw [h8]
    have h9 : |a| / ((1 : ℝ) + |a| + |b|) ≤ |a| / ((1 : ℝ) + |a|) := by
      have h10 : (1 : ℝ) + |a| + |b| ≥ (1 : ℝ) + |a| := by
        linarith [abs_nonneg b]
      have h11 : 0 < (1 : ℝ) + |a| := by
        linarith [abs_nonneg a]
      have h12 : 0 ≤ |a| := abs_nonneg a
      apply (div_le_div_iff (by linarith) (by linarith)).mpr
      nlinarith
    have h10 : |b| / ((1 : ℝ) + |a| + |b|) ≤ |b| / ((1 : ℝ) + |b|) := by
      have h11 : (1 : ℝ) + |a| + |b| ≥ (1 : ℝ) + |b| := by
        linarith [abs_nonneg a]
      have h12 : 0 < (1 : ℝ) + |b| := by
        linarith [abs_nonneg b]
      have h13 : 0 ≤ |b| := abs_nonneg b
      apply (div_le_div_iff (by linarith) (by linarith)).mpr
      nlinarith
    linarith
  linarith [h2, h3]