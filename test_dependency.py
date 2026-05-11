import os
import sys

# Đảm bảo import được betazero
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from betazero.core.nodes import ProofState
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
from betazero.utils.lean_cmd import build_theorem
import re

def get_allowed_vars(skeleton_code: str) -> set[str]:
    pattern = r"(?:have|let)\s+([a-zA-Z0-9_]+).*?:=\s*sorry"
    return set(re.findall(pattern, skeleton_code))

def run_test():
    state = ProofState(
        context="n : ℕ\nh : n > 0",
        goal="n + 1 > 1",
        header="import Mathlib\n"
    )

    print("==================================================")
    print("TEST 1: SKELETON CHỨA HAVE RÁC VÀ NAKED SORRY Ở MAIN GOAL")
    print("=> Tình huống: Chưa có subgoal nào được giải")
    skeleton_1 = "have h_unused : n > 0 := sorry\nsorry"
    # Cả hai đều chưa được giải (None)
    stitched_1 = ProofStitcher.stitch(skeleton_1, [None, None])
    print(f"\n[Stitched Code]\n{stitched_1}")
    
    full_1 = build_theorem(state, stitched_1)
    expr_1 = get_lean_expr_tree(full_1)
    if expr_1:
        allowed_1 = get_allowed_vars(skeleton_1)
        cls_1 = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(expr_1[-1]["expr_value_tree"], allowed_vars=allowed_1)
        print(f"\n[Classification]: {cls_1}")
        # Mong đợi: h_unused -> malignant, MAIN_GOAL -> core_failed

    print("\n==================================================")
    print("TEST 2: SKELETON CÓ NAKED SORRY, NHƯNG MAIN GOAL ĐÃ ĐƯỢC GIẢI")
    print("=> Tình huống: Subgoal 1 (have rác) vẫn sorry, Subgoal 2 (main goal) đã có tactic giải")
    # Tactic giải main goal: exact Nat.add_lt_add_right h 1
    stitched_2 = ProofStitcher.stitch(skeleton_1, [None, "exact Nat.add_lt_add_right h 1"])
    print(f"\n[Stitched Code]\n{stitched_2}")
    
    full_2 = build_theorem(state, stitched_2)
    expr_2 = get_lean_expr_tree(full_2)
    if expr_2:
        allowed_2 = get_allowed_vars(skeleton_1)
        cls_2 = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(expr_2[-1]["expr_value_tree"], allowed_vars=allowed_2)
        print(f"\n[Classification]: {cls_2}")
        # Mong đợi: h_unused -> malignant, không có MAIN_GOAL trong core_failed -> skeleton sẽ SOLVED.

    print("\n==================================================")
    print("TEST 3: SKELETON CÓ SỬ DỤNG HAVE, CẢ HAI CÙNG SORRY")
    print("=> Tình huống: Main goal gọi 'exact h2', nhưng h2 lại chứa sorry")
    skeleton_3 = "have h_used : n > 0 := sorry\nhave h2 : n + 1 > 1 := sorry\nexact h2"
    stitched_3 = ProofStitcher.stitch(skeleton_3, [None, None])
    print(f"\n[Stitched Code]\n{stitched_3}")
    
    full_3 = build_theorem(state, stitched_3)
    expr_3 = get_lean_expr_tree(full_3)
    if expr_3:
        allowed_3 = get_allowed_vars(skeleton_3)
        cls_3 = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(expr_3[-1]["expr_value_tree"], allowed_vars=allowed_3)
        print(f"\n[Classification]: {cls_3}")
        # Mong đợi: h2 được dùng -> core_failed chứa 'h2'. h_used không được dùng -> malignant.

    print("\n==================================================")
    print("TEST 4: SKELETON CÓ SỬ DỤNG HAVE ĐỂ GIẢI GOAL, HAVE ĐÃ ĐƯỢC GIẢI")
    print("=> Tình huống: h2 đã được giải, h_used vẫn sorry nhưng không được dùng.")
    stitched_4 = ProofStitcher.stitch(skeleton_3, [None, "exact Nat.add_lt_add_right h 1"])
    print(f"\n[Stitched Code]\n{stitched_4}")
    
    full_4 = build_theorem(state, stitched_4)
    expr_4 = get_lean_expr_tree(full_4)
    if expr_4:
        allowed_4 = get_allowed_vars(skeleton_3)
        cls_4 = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(expr_4[-1]["expr_value_tree"], allowed_vars=allowed_4)
        print(f"\n[Classification]: {cls_4}")
        # Mong đợi: h2 -> core_solved. h_used -> malignant. core_failed RỖNG -> skeleton SOLVED.

    print("\n==================================================")
    print("TEST 5: NAKED SORRY ĐƯỢC GIẢI BẰNG TACTIC NHIỀU DÒNG")
    print("=> Tình huống: Subgoal cuối cùng (sorry) được thay bằng một block 'by' nhiều dòng")
    
    multi_line_tactic = """
  have h_same : n > 0 := h
  exact Nat.add_lt_add_right h_same 1"""
    
    stitched_5 = ProofStitcher.stitch(skeleton_1, [None, multi_line_tactic])
    print(f"\n[Stitched Code]\n{stitched_5}")
    
    full_5 = build_theorem(state, stitched_5)
    expr_5 = get_lean_expr_tree(full_5)
    if expr_5:
        allowed_5 = get_allowed_vars(skeleton_1)
        cls_5 = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(expr_5[-1]["expr_value_tree"], allowed_vars=allowed_5)
        print(f"\n[Classification]: {cls_5}")
        # Mong đợi: core_solved chứa MAIN_GOAL. h_unused -> malignant.

if __name__ == '__main__':
    run_test()
