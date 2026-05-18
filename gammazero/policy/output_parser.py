import re
import textwrap

_LEAN_HEADER = re.compile(r"(?is)\b(theorem|lemma|example|def)\b")
_PROOF_DIVIDER = re.compile(r"(?is):=\s*by|(?<=\s)by(?=\s)")
_LEAN_FENCE = re.compile(r"```lean4\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)


def strip_lean_comments(code: str) -> str:
    return re.sub(r"/-(?:.|\n)*?-/|--.*", "", code)


def _final_lean_block(raw: str) -> str:
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
    code_no_comments = strip_lean_comments(code_block)
    
    # Only reject if '...' is used as a standalone placeholder (lazy model behavior)
    # Allows '...' in math comments or within valid Lean syntax if any.
    if re.search(r"^\s*\.\.\.\s*$", code_no_comments, re.MULTILINE):
        return ""

    return code_block


def get_lean_code(raw: str, *, allow_body: bool = False) -> str:
    """
    Only passes through fully valid Lean actions.
    Giữ nguyên 100% khoảng trắng và lề gốc của proof body.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""
    code_no_comments = strip_lean_comments(code_block)

    # Require header and proof divider unless this caller accepts a raw tactic
    # body.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        if not allow_body:
            return ""
        body_clean = code_no_comments.strip()
        if not body_clean or body_clean == "sorry" or "<|im_" in code_block:
            return ""
        return textwrap.dedent(code_block).strip("\n")

    divider_match = _PROOF_DIVIDER.search(code_block)
    
    # Lấy từ vị trí ngay sau chữ 'by' trở đi, giữ nguyên mọi dấu \n và space
    proof_body = code_block[divider_match.end():]

    # Inductive bias: Chỉ cho phép code có nội dung thực sự (không chỉ toàn comment/sorry).
    body_clean = re.sub(r"/-(?:.|\n)*?-/|--.*", "", proof_body).strip()
    if not body_clean or body_clean == "sorry" or "<|im_" in proof_body:
        return ""

    return proof_body


def _strip_leading_by(proof: str) -> str:
    """Return the tactic body inside an optional leading `by` wrapper."""
    proof = textwrap.dedent(proof).strip("\n")
    match = re.match(r"^\s*by(?P<tail>[ \t]*\n|[ \t]+|$)", proof)
    if not match:
        return proof

    body = proof[match.end():]
    # The old regex used `\s+`, which consumed the indentation of the first
    # tactic line after `by\n`. That turned aligned tactic blocks into
    # accidental nested blocks when stitched back into the scaffold.
    return textwrap.dedent(body).strip("\n")


def get_subgoal_tactic_code(raw: str, skeleton_code: str, target_child_index: int) -> str:
    """Extract only the replacement proof for one skeleton `sorry`.

    The subgoal tactic prompt asks the model to return the whole parent scaffold
    so Lean elaborates coercions in the same context. Search still stores and
    stitches only the proof replacing the target placeholder.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""

    # Backward-compatible fallback for older body-only responses.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return get_lean_code(raw, allow_body=True)

    divider_match = _PROOF_DIVIDER.search(code_block)
    if not divider_match:
        return ""

    proof_body = textwrap.dedent(code_block[divider_match.end():]).strip("\n")
    skeleton_body = textwrap.dedent(skeleton_code or "").strip("\n")
    parts = re.split(r"\bsorry\b", skeleton_body)
    sorry_count = len(parts) - 1
    if sorry_count <= 0 or target_child_index < 0 or target_child_index >= sorry_count:
        return ""

    pattern = re.escape(parts[0])
    for i in range(sorry_count):
        if i == target_child_index:
            pattern += r"(?P<replacement>.*?)"
        else:
            pattern += r"(?:by\s+admit|admit)"
        pattern += re.escape(parts[i + 1])

    match = re.search(pattern, proof_body, flags=re.DOTALL)
    if not match:
        return ""

    replacement = _strip_leading_by(match.group("replacement"))
    replacement_clean = strip_lean_comments(replacement).strip()
    if not replacement_clean or replacement_clean in {"sorry", "admit"}:
        return ""
    return replacement


def get_subgoal_skeleton_code(raw: str, skeleton_code: str, target_child_index: int) -> str:
    """Extract the mini-skeleton replacing one parent-skeleton `sorry`.

    Unlike tactic extraction, a skeleton replacement is allowed to contain new
    `sorry` leaves. Sibling placeholders in the returned parent scaffold must
    remain as `admit`.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""

    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return get_lean_code(raw, allow_body=True)

    divider_match = _PROOF_DIVIDER.search(code_block)
    if not divider_match:
        return ""

    proof_body = textwrap.dedent(code_block[divider_match.end():]).strip("\n")
    skeleton_body = textwrap.dedent(skeleton_code or "").strip("\n")
    parts = re.split(r"\bsorry\b", skeleton_body)
    sorry_count = len(parts) - 1
    if sorry_count <= 0 or target_child_index < 0 or target_child_index >= sorry_count:
        return ""

    pattern = re.escape(parts[0])
    for i in range(sorry_count):
        if i == target_child_index:
            pattern += r"(?P<replacement>.*?)"
        else:
            pattern += r"(?:by\s+admit|admit)"
        pattern += re.escape(parts[i + 1])

    match = re.search(pattern, proof_body, flags=re.DOTALL)
    if not match:
        return ""

    replacement = _strip_leading_by(match.group("replacement"))
    replacement_clean = strip_lean_comments(replacement).strip()
    if not replacement_clean or replacement_clean == "admit":
        return ""
    return replacement
