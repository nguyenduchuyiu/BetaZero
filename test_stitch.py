from betazero.search.sorrifier.stitcher import ProofStitcher
skeleton = '''  have h_expr : ∀ x ∈ S, f x = 30 - x := sorry
  rw [hS]
'''
child = '''by intro x hx
  rw [hS] at hx'''
print(ProofStitcher.stitch(skeleton, [child]))

skeleton2 = '''  constructor
  · sorry
'''
child2 = '''rw [h_expr]
norm_num'''
print(ProofStitcher.stitch(skeleton2, [child2]))
