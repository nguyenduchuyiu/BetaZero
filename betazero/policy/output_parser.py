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
    # Reject if there is an unclosed block at the end (indicates truncation).
    last_open = t.rfind("```lean4")
    if last_open != -1:
        last_close = t.rfind("```", last_open + 7)
        if last_close == -1:
            return ""

    fences = re.findall(r"```lean4\s+(.*?)\s+```", t, re.DOTALL | re.IGNORECASE)
    if not fences:
        return ""
    
    code_block = fences[-1]

    # Reject if it contains placeholders like '...' which indicate incomplete logic.
    if "..." in code_block:
        return ""

    # Require header and proof divider.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return ""

    divider_match = _PROOF_DIVIDER.search(code_block)
    
    # Lấy từ vị trí ngay sau chữ 'by' trở đi, giữ nguyên mọi dấu \n và space
    proof_body = code_block[divider_match.end():]

    # Inductive bias: Chỉ cho phép code có nội dung thực sự (không chỉ toàn comment/sorry).
    body_clean = re.sub(r"/-(?:.|\n)*?-/|--.*", "", proof_body).strip()
    if not body_clean or body_clean == "sorry" or "<|im_" in proof_body:
        return ""

    return proof_body
    