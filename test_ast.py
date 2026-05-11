from betazero.env.lean_env import LeanEnv
from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER

print("Starting test...")
code = """
open BigOperators Nat Real Topology
theorem my_theorem (f : ℤ → ℤ) (h₁ : ∀ (x : ℤ), f x + f (x - 1) = x ^ 2) (h₂ : f 19 = 94) (h20 : f 20 = 306) (h21 : f 21 = 135) (h22 : f 22 = 349) (h23 : f 23 = 180) (h24 : f 24 = 396) (h25 : f 25 = 229) (h26 : f 26 = 447) (h27 : f 27 = 282) (h28 : f 28 = 502) (h29 : f 29 = 339) (h30 : f 30 = 561) (h31 : f 31 = 400) : f 32 = 624 := by
  have h_eq : f 32 + f 31 = 1024 := by sorry
  have h_final : f 32 = 624 := by
    linarith
  exact h_final
"""

try:
    lean = LeanEnv()
    print("LeanEnv created.")
    ast = lean.get_kernel_expr(code, "my_theorem")
    print(f"AST is None? {ast is None}")
    if ast is not None:
        res = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(ast, {"h_eq", "h_final"})
        print("Result:", res)
    else:
        print("Failed to get AST")
except Exception as e:
    print("Exception:", e)
