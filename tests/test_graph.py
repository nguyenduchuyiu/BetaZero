import pytest
import re
from betazero.core.nodes import ProofState, Action
from betazero.search.graph import ANDORGraph

# ==========================================
# TEST 1: CHỐNG VÒNG LẶP VÔ HẠN (CIRCULAR DEPENDENCY)
# ==========================================
def test_circular_dependency_handling():
    """
    Kịch bản: LLM ngáo, đẻ ra một subgoal có goal y hệt root.
    Kỳ vọng: Đồ thị phát hiện vòng lặp qua `visiting` set, cắt đứt đệ quy,
    không bị văng lỗi RecursionError (Stack Overflow) và fallback trả về sorry.
    """
    root = ProofState(context="", goal="f 94 % 1000 = 561", header="")
    graph = ANDORGraph(root)

    # Action này có child trỏ thẳng về root (tạo vòng lặp)
    action_loop = Action(
        action_type="skeleton",
        content="have h := sorry; exact h",
        extracted_code="have h := sorry\nexact h",
        children=(root,)  # <-- CIRCULAR HERE
    )
    
    # Ép nó thành SOLVED (giả lập lỗi False Positive của Lean kernel trước khi dọn dẹp)
    graph.expand(root, action_loop, r_env=1.0)
    graph.set_skeleton_override(action_loop, True)

    # Nếu không có cơ chế `visiting`, dòng này sẽ gây crash toàn bộ hệ thống
    proof = graph.extract_proof_code(root)
    
    assert proof is not None
    assert "sorry" in proof, "Phải fallback về sorry do vòng lặp bị ngắt"


# ==========================================
# TEST 2: ƯU TIÊN PROOF SẠCH SẼ (CLEAN PROOF PRIORITY)
# ==========================================
def test_prioritize_clean_proof():
    """
    Kịch bản: Một state có 2 đường (Action) cùng được đánh dấu SOLVED.
    Đường 1: Có chứa 'sorry' (Mô hình lươn lẹo, Lean bị lừa).
    Đường 2: Proof sạch 100% không chứa sorry.
    Kỳ vọng: extract_proof_code phải khôn ngoan bóp chết đường 1 và chọn đường 2.
    """
    root = ProofState(context="", goal="1 + 1 = 2", header="")
    graph = ANDORGraph(root)

    # Đường 1: Bẩn (nhưng vẫn được tính là SOLVED)
    action_dirty = Action(
        action_type="tactic",
        content="exact sorry",
        extracted_code="exact sorry",
        children=()
    )
    graph.expand(root, action_dirty, r_env=1.0, tactic_status="SOLVED")

    # Đường 2: Sạch
    action_clean = Action(
        action_type="tactic",
        content="rfl",
        extracted_code="rfl",
        children=()
    )
    graph.expand(root, action_clean, r_env=1.0, tactic_status="SOLVED")

    proof = graph.extract_proof_code(root)
    
    assert proof == "rfl", "Phải ưu tiên chọn proof sạch (rfl) thay vì proof chứa sorry"


# ==========================================
# TEST 3: FALLBACK VỀ PROOF BẨN KHI HẾT CÁCH
# ==========================================
def test_fallback_to_sorry_when_no_clean_proof():
    """
    Kịch bản: State có action SOLVED, nhưng TẤT CẢ các action đều chứa sorry.
    Kỳ vọng: Hệ thống không trả về None, mà lấy cái proof bẩn đầu tiên để 
    ít nhất vẫn có code ghép vào cho Lean chạy chấm điểm.
    """
    root = ProofState(context="x : Nat", goal="x = x", header="")
    graph = ANDORGraph(root)

    action_dirty = Action(
        action_type="tactic",
        content="exact sorry",
        extracted_code="-- proof starts\nexact sorry",
        children=()
    )
    graph.expand(root, action_dirty, r_env=1.0, tactic_status="SOLVED")

    proof = graph.extract_proof_code(root)
    
    assert proof is not None
    assert "sorry" in proof, "Phải fallback trả về proof chứa sorry"


# ==========================================
# TEST 4: BỎ QUA COMMENT KHI TÌM SORRY
# ==========================================
def test_ignore_sorry_inside_comments():
    """
    Kịch bản: Code hoàn toàn sạch, chữ 'sorry' chỉ nằm trong comment do 
    ProofPruner (máy xén tỉa) comment lại.
    Kỳ vọng: Hệ thống hiểu đây là proof sạch, không bắt nhầm.
    """
    root = ProofState(context="", goal="A", header="")
    graph = ANDORGraph(root)

    clean_code_with_comment = (
        "-- [PRUNED] have h : B := sorry\n"
        "/- I removed a sorry here -/\n"
        "exact A_proof"
    )
    
    action_commented = Action(
        action_type="tactic",
        content="...",
        extracted_code=clean_code_with_comment,
        children=()
    )
    graph.expand(root, action_commented, r_env=1.0, tactic_status="SOLVED")

    proof = graph.extract_proof_code(root)
    
    assert proof == clean_code_with_comment
    # Tự test lại logic regex tích hợp trong ANDORGraph
    clean_check = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", proof)
    assert not re.search(r'\bsorry\b', clean_check), "Regex phải bỏ qua sorry trong comment"


# ==========================================
# TEST 5: SKELETON GỌI TACTIC SẠCH VÀ BẨN
# ==========================================
def test_skeleton_stitching_with_clean_and_dirty_children():
    """
    Kịch bản: Skeleton chia 2 nhánh. Nhánh 1 giải sạch, Nhánh 2 bí đắp sorry.
    Kỳ vọng: Bản stitch cuối cùng phải chứa sorry do kế thừa từ nhánh 2.
    """
    root = ProofState(context="", goal="C", header="")
    graph = ANDORGraph(root)

    child_1 = ProofState(context="", goal="A", header="")
    child_2 = ProofState(context="", goal="B", header="")

    skeleton_action = Action(
        action_type="skeleton",
        content="have h1 := sorry\nhave h2 := sorry\nexact h2",
        extracted_code="have h1 := sorry\nhave h2 := sorry\nexact h2",
        children=(child_1, child_2)
    )
    graph.expand(root, skeleton_action, r_env=1.0)
    
    # Giải sạch nhánh 1
    t1 = Action(action_type="tactic", content="rfl", extracted_code="rfl", children=())
    graph.expand(child_1, t1, r_env=1.0, tactic_status="SOLVED")

    # Bí ở nhánh 2
    t2 = Action(action_type="tactic", content="sorry", extracted_code="sorry", children=())
    graph.expand(child_2, t2, r_env=0.0, tactic_status="SOLVED") # Cố tình dán SOLVED để test

    graph.set_skeleton_override(skeleton_action, True)

    proof = graph.extract_proof_code(root)
    
    assert "rfl" in proof, "Phải ghép được proof của nhánh 1"
    assert "sorry" in proof, "Phải dính sorry từ nhánh 2"

if __name__ == "__main__":
    pytest.main(["-v", __file__])