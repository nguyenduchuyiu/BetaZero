import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from betazero.logic.reward import RewardCalculator
from betazero.logic.sorrifier import Sorrifier
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
    original_code = """example (a b : Nat) : a + b = b + a := by
  rfl
  rw [Nat.add_comm]
  done
  skip"""

    print("1. Chạy mock Auto-Sorrifier...")
    patched_code = mock_sorrify(original_code)

    print("2. Chạy Lean Verifier thật trên patched code...")
    # Lưu ý: Hàm verify_lean_code phải được import đúng từ môi trường của bạn
    request_ids = verifier.submit_all_request([{'code': patched_code, 'timeout': 10}])
    verify_result = verifier.get_all_request_outputs(request_ids)[0]
    print("3. Tính toán r_env...")
    reward = calc.r_env(original_code, patched_code, verify_result)

    print("\n=== KẾT QUẢ ===")
    print("--- Original Code ---")
    print(original_code)
    print("\n--- Patched Code ---")
    print(patched_code)
    print("\n--- Verify Errors/Warnings ---")
    print(f"Sorries: {len(verify_result.get('sorries', []))}")
    print(f"Warnings: {[w.get('data') for w in verify_result.get('warnings', [])]}")
    
    print(f"\n=> Điểm r_env cuối cùng: {reward:.4f}")