import pytest
import re
from betazero.core.nodes import ProofState, Action
from betazero.search.graph import ANDORGraph

def test_stitch_and_garbage_collection():
    """
    Kịch bản: 1 Root -> 1 Skeleton -> 3 Subgoals (Core, Benign, Malignant).
    - Core: Giải bằng tactic sạch, được gọi ở dòng chốt hạ.
    - Benign: Giải bằng tactic sạch, nhưng không được gọi.
    - Malignant: Bí, phải đắp sorry, không được gọi.
    
    Kỳ vọng sau khi stitch và extract:
    - h_core: Còn nguyên vẹn (Được đắp code sạch).
    - h_beg: Bị comment `-- [PRUNED]`.
    - h_mag: Bị comment `-- [PRUNED]` (Che luôn cả chữ sorry của nó).
    - Toàn bộ đoạn code hoàn toàn SẠCH BÓNG sorry đối với trình biên dịch Lean.
    """
    
    # 1. Khởi tạo Root
    root = ProofState(context="", goal="MainGoal", header="")
    graph = ANDORGraph(root)

    # 2. Tạo 3 Subgoals con
    child_core = ProofState(context="", goal="CoreProp", header="")
    child_beg  = ProofState(context="", goal="BegProp", header="")
    child_mag  = ProofState(context="", goal="MagProp", header="")

    # 3. Khởi tạo Skeleton chia làm 3 nhánh
    # Lưu ý: Lệnh chốt hạ chỉ xài h_core
    skeleton_code = (
        "have h_core : CoreProp := sorry\n"
        "have h_beg : BegProp := sorry\n"
        "have h_mag : MagProp := sorry\n"
        "exact h_core"
    )
    action_skel = Action(
        action_type="skeleton",
        content="...",
        extracted_code=skeleton_code,
        children=(child_core, child_beg, child_mag)
    )
    graph.expand(root, action_skel, r_env=1.0)

    # 4. Giải quyết nhánh Core (Tuyệt đối sạch)
    t_core = Action(action_type="tactic", content="", extracted_code="exact X", children=())
    graph.expand(child_core, t_core, r_env=1.0, tactic_status="SOLVED")

# 5. Giải quyết nhánh Benign (Sạch, nhưng vô dụng cho kết luận)
    t_beg = Action(action_type="tactic", content="", extracted_code="intro x\n  rfl", children=())
    graph.expand(child_beg, t_beg, r_env=1.0, tactic_status="SOLVED")

    # 6. Giải quyết nhánh Malignant (Bẩn, chứa sorry, vô dụng)
    t_mag = Action(action_type="tactic", content="", extracted_code="intro y\n  exact sorry", children=())
    graph.expand(child_mag, t_mag, r_env=0.0, tactic_status="SOLVED")
    # 7. GIẢ LẬP MODULE DEPENDENCY REWARD ASSIGNER
    # Thay vì gọi Lean thực tế, ta bơm thẳng kết quả phân loại rác cho Action này
    # (Đúng với logic bạn vừa thêm vào hàm assign)
    graph.set_garbage_vars(action_skel, ["h_beg", "h_mag"])
    graph.set_skeleton_override(action_skel, True)

    # 8. THỰC THI STITCH VÀ EXTRACT
    proof = graph.extract_proof_code(root)

    print("\n\n" + "="*40)
    print("MÃ CHỨNG MINH LEAN SAU KHI STITCH & PRUNE:")
    print("-" * 40)
    print(proof)
    print("="*40 + "\n")

    # 9. CÁC BƯỚC ASSERT (KIỂM CHỨNG)
    assert proof is not None, "Proof không được phép None"

    # Core phải sống sót và được đắp code
    assert "have h_core : CoreProp := by exact X" in proof or "have h_core : CoreProp := exact X" in proof
    assert "-- [PRUNED] have h_core" not in proof, "Core tuyệt đối không được bị comment!"

    # Benign phải bị chém
    assert "-- [PRUNED] have h_beg : BegProp :=" in proof, "Benign phải bị dọn rác"
    assert "-- [PRUNED]   rfl" in proof or "-- [PRUNED] rfl" in proof, "Code của Benign cũng phải bị comment theo"

    # Malignant phải bị chém
    assert "-- [PRUNED] have h_mag : MagProp :=" in proof, "Malignant phải bị dọn rác"
    assert "-- [PRUNED]   exact sorry" in proof or "-- [PRUNED] exact sorry" in proof, "Chữ sorry của Malignant phải bị comment giấu đi"

    # CÚ CHỐT: Máy quét của Lean (Mô phỏng) không được nhìn thấy chữ sorry nào đang "thả rông"
    clean_check = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", proof)
    assert not re.search(r'\bsorry\b', clean_check), "Báo động: Vẫn còn chữ sorry lọt ra ngoài, Lean sẽ báo False!"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])