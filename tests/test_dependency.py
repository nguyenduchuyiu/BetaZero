import json
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import ExprDependencyAnalyzer
from betazero.search.reward.calculator import RewardCalculator

def test_real_kernel_dependency():
    print("--- 🚀 BẮT ĐẦU TEST DEPENDENCY VỚI KERNEL THẬT ---")

    # ĐOẠN CODE LEAN "CÀI BẪY"
    # h_core: giải xịn
    # h_benign: giải xịn nhưng rác
    # h_malignant: rác + sorry
    lean_code = """
theorem my_theorem (a b : Nat) (h_used : a = b) : a = b := by
  have h_core : a = b := by 
    exact h_used
  have h_benign : 1 = 1 := by 
    rfl
  have h_malignant : 2 = 2 := by 
    sorry
  exact h_core
"""

    print("[1] Đang gọi Lean Daemon để lấy EXPR Tree...")
    results = get_lean_expr_tree(lean_code)
    
    if not results:
        print("❌ Lỗi: Không lấy được Expr Tree. Kiểm tra lại Daemon!")
        return

    root_expr = results[-1].get("expr_value_tree")
    analyzer = ExprDependencyAnalyzer()

    print("\n[2] PHÂN LOẠI SUBGOALS (Skeleton Analysis):")
    classification = analyzer.classify_skeleton_subgoals(root_expr)
    print(json.dumps(classification, indent=2))
    
    print("\n[3] TÍNH TOÁN r_dep THEO LOGIC MỚI:")
    calculator = RewardCalculator()
    
    if len(classification.get("core_failed", [])) > 0:
        r_dep = 0.0
        print("👉 core_failed > 0 -> r_dep = 0.0")
    else:
        mapped_analysis = {
            "core": classification.get("core_solved", []),
            "benign": classification.get("benign", []),
            "malignant": classification.get("malignant", [])
        }
        r_dep = calculator.r_dep(mapped_analysis)
        print("👉 Skeleton hợp lệ (không có core_failed). Tính r_dep:")
        print(f"   - Core: {len(mapped_analysis['core'])}")
        print(f"   - Benign: {len(mapped_analysis['benign'])}")
        print(f"   - Malignant: {len(mapped_analysis['malignant'])}")
        print(f"👉 r_dep = {r_dep}")

if __name__ == "__main__":
    test_real_kernel_dependency()