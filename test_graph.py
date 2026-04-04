import sys
import os
from unittest.mock import MagicMock

# Đảm bảo Python nhận diện được package betazero
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORT HÀNG REAL TỪ CODEBASE CỦA MÀY
from betazero.core.nodes import ProofState, Action
from betazero.search.graph.and_or_graph import ANDORGraph
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.sorrifier.stitcher import ProofStitcher

def run_real_test():
    print("--- BẮT ĐẦU TEST BẰNG CODE THẬT ---")
    
    # 1. Khởi tạo State và Action (Hàng Real)
    S_root = ProofState(context="", goal="A AND B", header="")
    S_core = ProofState(context="", goal="A", header="")
    S_junk = ProofState(context="", goal="C", header="") 

    # Skeleton với 2 sorry
    A_skel = Action(
        action_type="skeleton", 
        content="have h1 : A := sorry\nhave h_junk : C := sorry\nexact h1", 
        children=(S_core, S_junk)
    )
    
    # Dùng ANDORGraph THẬT
    graph = ANDORGraph(S_root)
    graph.expand(S_root, A_skel, r_env=0.5)

    # Giải subgoal Core thành công
    A_core_solve = Action("tactic", "exact a_proof", children=())
    graph.expand(S_core, A_core_solve, r_env=1.0, tactic_status="SOLVED")

    # Giải subgoal Junk thất bại
    A_junk_fail = Action("tactic", "exact wrong", children=())
    graph.expand(S_junk, A_junk_fail, r_env=0.1, tactic_status="FAILED")

    print("\n[Trạng thái trước Assigner (Đọc từ Graph thật)]")
    print(f"S_core solved? {graph.is_solved(S_core)}")
    print(f"S_junk solved? {graph.is_solved(S_junk)}")
    print(f"Skeleton solved? {graph.is_solved(A_skel)}")

    # 2. Chạy logic của Assigner
    print("\n--- CHẠY LOGIC ASSIGNER ---")
    for action, state in graph.parent_items():
        if action.action_type == "skeleton":
            # Test hàm extract_proof_code thật của mày
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            
            # Test hàm Stitch thật
            stitched_code = ProofStitcher.stitch(action.content, child_proofs)
            print(f"Stitched Code từ hàm thật:\n{stitched_code}")

            # MOCK MỖI CÁI LEAN KERNEL (Vì ta đéo muốn bật server lúc test graph)
            mock_dep_analysis = {"core_solved": ["h1"], "core_failed": [], "malignant": ["h_junk"]}
            
            # Test RewardCalculator thật
            calc = RewardCalculator(W_c=1.0, W_b=0.0, W_m=-0.5)
            r_dep_score = calc.r_dep(mock_dep_analysis)
            graph.set_r_dep(action, r_dep_score)
            print(f"Calculated r_dep = {r_dep_score}")

            if not mock_dep_analysis.get("core_failed"):
                print(">> Kích hoạt override trên Graph thật!")
                graph.set_skeleton_override(action, True)

    # 3. Test thuật toán Backup thật
    q_cache = graph.backup()

    print("\n--- KẾT QUẢ LAN TRUYỀN (HÀNG REAL) ---")
    print(f"Q(A_skel) = {q_cache.get(A_skel)}")
    # V_cache mày ẩn ở bên trong backup rồi, nhưng Q_cache đúng là V_root cũng sẽ đúng!

if __name__ == "__main__":
    run_real_test()