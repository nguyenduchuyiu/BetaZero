from betazero.search.sorrifier.stitcher import ProofStitcher

skeleton = """  have ha : a ≠ 0 := h₀.left
  have hb : b ≠ 0 := h₀.right
  have h_ab_nonzero : a * b ≠ 0 := sorry
  have h_frac_sum : a / b + b / a = (a ^ 2 + b ^ 2) / (a * b) := sorry
  have h_target_rewrite : a / b + b / a - a * b = ((a ^ 2 + b ^ 2) / (a * b)) - a * b := sorry
  have h_main : ((a ^ 2 + b ^ 2) / (a * b)) - a * b = 2 := sorry
  have h_final : a / b + b / a - a * b = 2 := sorry
  exact h_final"""

child_proofs = [
    "exact mul_ne_zero ha hb", # No 'by' here
    "by field_simp [ha, hb]",
    "by rw [h_frac_sum]",
    "by\n  field_simp [h_ab_nonzero]\n  nlinarith [h₁]",
    "by rw [h_target_rewrite, h_main]"
]

stitched = ProofStitcher.stitch(skeleton, child_proofs)
print("--- STITCHED CODE ---")
print(stitched)
print("---------------------")
