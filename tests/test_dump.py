import json
from betazero.env.expr_parser import get_lean_expr_tree

code1 = """
theorem my_theorem (a b : Nat) (h_used : a = b) : a = b := by
  have h_core : a = b := by 
    exact h_used
  have h_benign : 1 = 1 := by 
    rfl
  have h_malignant : 2 = 2 := by 
    sorry
  exact h_core
"""

code2 = """import Mathlib
theorem my_theorem (a b : Nat) (h_used : a = b) : a = b := by
  have h_core : a = b := by exact h_used
  have h_benign : 1 = 1 := by rfl
  have h_malignant : 2 = 2 := by sorry
  exact h_core
"""

print("Running code 1...")
res1 = get_lean_expr_tree(code1)
name1 = res1[-1].get("theorem_name", "UNKNOWN") if res1 else "NONE"
print(f"Code 1 Last Theorem: {name1}")

print("Running code 2...")
res2 = get_lean_expr_tree(code2)
name2 = res2[-1].get("theorem_name", "UNKNOWN") if res2 else "NONE"
print(f"Code 2 Last Theorem: {name2}")

