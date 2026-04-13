import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from betazero.utils.lean_parse import parse_proof_state

# Giả lập output từ Lean REPL với một giả thiết bị xuống dòng
repl_output = """x : ℝ
h₀ : 0 < x
h_mul_clear :
  3 * Real.log 2 * Real.log (Real.log x / (3 * Real.log 2)) =
  Real.log 2 * Real.log (Real.log x / Real.log 2)
⊢ Real.log ((t / 3) ^ 3) = Real.log t"""

parsed = parse_proof_state(repl_output)
print("--- CONTEXT ---")
print(parsed.context)
print("--- GOAL ---")
print(parsed.goal)

# Kiểm tra xem h_mul_clear có chứa nội dung sau dấu : không
if "h_mul_clear :" in parsed.context and "=" in parsed.context:
    print("\n✅ TEST PASSED: Hypothesis merged successfully.")
else:
    print("\n❌ TEST FAILED: Hypothesis still truncated.")
