import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from betazero.search.reward import RewardCalculator
from betazero.search.sorrifier import Sorrifier
from betazero.env.lean_verifier import Lean4ServerScheduler

verifier = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, name="auto_sorrifier_cli")


def mock_sorrify(code: str) -> str:
    """Mock Auto-Sorrifier: Thay thế dòng code lỗi bằng sorry."""
    sorrifier = Sorrifier(verifier)
    fixed_code = sorrifier.fix_code(code)
    return fixed_code

if __name__ == "__main__":
    calc = RewardCalculator()

    # Code gốc:
    # 1. Có 1 lệnh hợp lệ (rw)
    # 2. Có 1 lệnh lỗi (exact xyz...)
      # 3. Có 2 lệnh rác linter sẽ báo unused/does nothing (skip)
    import os
    import json
    import sys

    # Add the project root to path
    sys.path.append("/workspace/npthai/BetaZero")

    from betazero.env.lean_verifier import PersistentLeanWorker
    from betazero.utils.lean_cmd import build_theorem
    from betazero.core import ProofState

    # Mimic the state in action_175
    state = ProofState(
        context="a b : ℝ\nh₀ : logb 8 a + logb 4 (b ^ 2) = 5\nh₁ : logb 8 b + logb 4 (a ^ 2) = 7\nh_log8_4 : logb 8 4 = 2 / 3\nh_eq1' : logb 8 (b ^ 2) = 2 * logb 8 b / logb 8 4\nh_eq2' : logb 8 (a ^ 2) = 2 * logb 8 a / logb 8 4\nh_rewrite0 : logb 8 a + 2 * logb 8 b / logb 8 4 = 5\nh_rewrite1 : logb 8 b + 2 * logb 8 a / logb 8 4 = 7\nh_subst0 : logb 8 a + 3 * logb 8 b = 5\nh_subst1 : logb 8 b + 3 * logb 8 a = 7\nh_log_a : logb 8 a = 2",
        goal="logb 8 b = 1",
        header="open BigOperators Nat Real Topology"
    )
    code = "\n  linarith"

    full_code = build_theorem(state, code)

    print("1. Chạy mock Auto-Sorrifier...")
    patched_code = mock_sorrify(full_code)

    print("2. Chạy Lean Verifier thật trên patched code...")
    # Lưu ý: Hàm verify_lean_code phải được import đúng từ môi trường của bạn
    request_ids = verifier.submit_all_request([{'code': patched_code, 'timeout': 10}])
    verify_result = verifier.get_all_request_outputs(request_ids)[0]
    print("3. Tính toán r_env...")
    reward = calc.r_env(full_code, patched_code, verify_result)

    print("\n=== KẾT QUẢ ===")
    print("--- Original Code ---")
    print(full_code)
    print("\n--- Patched Code ---")
    print(patched_code)
    print("\n--- Verify Errors/Warnings ---")
    print(f"Sorries: {len(verify_result.get('sorries', []))}")
    print(f"Warnings: {[w.get('data') for w in verify_result.get('warnings', [])]}")
    
    print(f"\n=> Điểm r_env cuối cùng: {reward:.4f}")