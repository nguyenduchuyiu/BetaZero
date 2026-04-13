import json
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import ExprDependencyAnalyzer

def test_real_kernel_dependency():
    print("--- 🚀 BẮT ĐẦU TEST DEPENDENCY VỚI KERNEL THẬT ---")

    # ĐOẠN CODE LEAN "CÀI BẪY"
    # h_used: có dùng, h_unused: vứt đi
    # h_core: giải xịn, h_junk: rác + sorry
    lean_code = """
theorem my_theorem (a b c : Nat) (h_used : a = b) (h_unused : b = c) : a = b := by
  have h_core : a = b := by 
    exact h_used
  have h_junk : 3 = 3 := by 
    sorry
  exact h_core
"""

    print("[1] Đang gọi Lean Daemon để lấy EXPR Tree...")
    # Lấy Expr Tree thật từ server của mày
    results = get_lean_expr_tree(lean_code)
    
    if not results:
        print("❌ Lỗi: Không lấy được Expr Tree. Kiểm tra lại Daemon!")
        return

    # Lấy cái theorem cuối cùng trong list results
    root_expr = results[-1].get("expr_value_tree")
    print("✅ Đã lấy được Expr Tree thành công.")
    print("Expr Value Tree:\n ", json.dumps(root_expr, indent=2))
    analyzer = ExprDependencyAnalyzer()

    print("\n[2] PHÂN TÍCH GIẢ THIẾT THỪA (Context Pruning):")
    unused = analyzer.get_unused_context_variables(root_expr)
    print(f"👉 Unused variables: {unused}")
    # KỲ VỌNG: Phải thấy 'h_unused', KHÔNG ĐƯỢC thấy 'h_used'

    print("\n[3] PHÂN LOẠI SUBGOALS (Skeleton Analysis):")
    classification = analyzer.classify_skeleton_subgoals(root_expr)
    print(json.dumps(classification, indent=2))
    # KỲ VỌNG: 
    # - 'h_core' nằm trong core_solved
    # - 'h_junk' nằm trong malignant (vì đéo ai dùng nó cả)

if __name__ == "__main__":
    test_real_kernel_dependency()