import re
import textwrap

_LEAN_HEADER = re.compile(r"(?is)\b(theorem|lemma|example|def)\b")
_PROOF_DIVIDER = re.compile(r"(?is):=\s*by|(?<=\s)by(?=\s)")

def get_lean_code(raw: str) -> str:
    """
    Only passes through fully valid Lean actions.
    Giữ nguyên 100% khoảng trắng và lề gốc của proof body.
    """
    t = raw.strip()

    # Reject if ChatML tokens are present.
    if "<|im_" in t:
        return ""

    # Capture last ```lean4 ... ``` code block.
    # Lưu ý: Giữ lại \s+ để tương thích với regex gốc của ông
    fences = re.findall(r"```lean4\s+(.*?)\s+```", t, re.DOTALL | re.IGNORECASE)
    if not fences:
        return ""
    
    # KHÔNG .strip() code_block vội, cứ để nguyên raw string
    code_block = fences[-1]

    # Require header and proof divider.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return ""

    divider_match = _PROOF_DIVIDER.search(code_block)
    
    # Lấy từ vị trí ngay sau chữ 'by' trở đi, giữ nguyên mọi dấu \n và space
    proof_body = code_block[divider_match.end():]

    if not proof_body or "<|im_" in proof_body:
        return ""

    return proof_body
    
if __name__ == "__main__":
    test = '''- After change: `(25 + 5) rows * (18 - 3) seats = 30 rows * 15 seats = 450 seats`.
    - Both arrangements result in 450 seats, confirming the solution is correct.

**Lean 4 Code:**

```lean4
theorem <ACTUAL_THEOREM_NAME> (<ACTUAL_VARIABLES>) : <ACTUAL_GOAL> := by
    -- Use the quadratic formula to solve the quadratic equation derived from the problem.
    have h : rows = 25 := by
        -- Apply the quadratic formula to the equation rows^2 + 5rows - 750 = 0.
        apply Eq.symm
        nlinarith [sq_nonneg (rows - 25), sq_nonneg (rows + 30)]
    -- Substitute the value of rows into the equation to find the number of seats.
    have h' : seats = 18 := by
        subst h
        nlinarith
    -- Verify the solution by checking the original and modified arrangements.
    subst_vars
    nlinarith
```'''
    print(get_lean_code(test))