import re
import textwrap

_LEAN_HEADER = re.compile(r"(?is)\b(theorem|lemma|example|def)\b")
_PROOF_DIVIDER = re.compile(r"(?is):=\s*by|(?<=\s)by(?=\s)")
_LEAN_FENCE = re.compile(r"```lean4\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)

def get_lean_code(raw: str) -> str:
    """
    Only passes through fully valid Lean actions.
    Giữ nguyên 100% khoảng trắng và lề gốc của proof body.
    """
    t = raw.strip()

    # Reject if ChatML tokens are present.
    if "<|im_" in t:
        return ""

    # Require the final Lean code block to be the end of the response. The
    # thinking section may contain draft Lean fences; only the final block is
    # treated as the answer.
    fences = list(_LEAN_FENCE.finditer(t))
    if not fences or fences[-1].end() != len(t):
        return ""

    code_block = fences[-1].group(1)

    # Remove comments before checking for placeholders like '...'
    code_no_comments = re.sub(r"/-(?:.|\n)*?-/|--.*", "", code_block)
    
    # Only reject if '...' is used as a standalone placeholder (lazy model behavior)
    # Allows '...' in math comments or within valid Lean syntax if any.
    if re.search(r"^\s*\.\.\.\s*$", code_no_comments, re.MULTILINE):
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
    
