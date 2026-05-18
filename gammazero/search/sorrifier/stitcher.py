"""Safe textual proof stitcher for filling skeleton subgoals."""

from __future__ import annotations
import re

class ProofStitcher:
    """Stitches child proof blocks into skeleton sorry placeholders."""

    @staticmethod
    def stitch(skeleton_code: str, child_proofs: list[str | None]) -> str:
        """
        Replaces each `sorry` in the skeleton with the corresponding child proof.
        If a child proof is None (FAILED), `sorry` remains.
        """
        # Split strictly by the word 'sorry'
        parts = re.split(r'\bsorry\b', skeleton_code)
        
        if len(parts) - 1 != len(child_proofs):
            # Fallback for LLM hallucination: mismatch between sorry count and children
            return skeleton_code

        stitched = parts[0]
        for i, proof in enumerate(child_proofs):
            if proof is not None:
                import textwrap
                
                lines = stitched.splitlines()
                last_line = lines[-1] if lines else ""
                
                # Normalize child proof indentation (remove common leading whitespace)
                clean_proof = textwrap.dedent(proof).strip("\n")
                
                # Check if we are filling an assignment ':='
                prefix = parts[i].rstrip()
                is_assignment = prefix.endswith(":=")
                
                if is_assignment:
                    base_indent = " " * (len(last_line) - len(last_line.lstrip()))
                    child_indent = base_indent + "  "
                    
                    if clean_proof.startswith("by\n"):
                        clean_proof = textwrap.dedent(clean_proof[3:]).strip("\n")
                    elif clean_proof.startswith("by "):
                        clean_proof = clean_proof[3:].strip("\n")
                        
                    proof_lines = clean_proof.splitlines()
                    indented_body = "\n".join(child_indent + l for l in proof_lines)
                    
                    if parts[i].endswith(" "):
                        indented_proof = "by\n" + indented_body
                    else:
                        indented_proof = " by\n" + indented_body
                else:
                    anchor_indent = " " * len(last_line)
                    proof_lines = clean_proof.splitlines()
                    indented_proof = "\n".join(
                        (anchor_indent + l if idx > 0 else l) for idx, l in enumerate(proof_lines)
                    )
                
                stitched += indented_proof
            else:
                stitched += "sorry"
                
            stitched += parts[i + 1]

        return stitched

    @staticmethod
    def prune_garbage(stitched_code: str, garbage_vars: list[str]) -> str:
        """
        Quét và comment lại toàn bộ các dòng khai báo biến rác.
        Hỗ trợ dọn dẹp các khối proof nhiều dòng dựa trên Indentation.
        """
        if not garbage_vars:
            return stitched_code

        lines = stitched_code.splitlines()
        out_lines = []
        
        # Regex tìm chính xác biến rác
        garbage_patterns = [re.compile(rf"^\s*(?:have|let)\s+{re.escape(var)}\b") for var in garbage_vars]

        skip_mode = False
        base_indent = 0 # Lưu lại độ thụt lề của chữ 'have'
        
        for line in lines:
            # 1. Phát hiện dòng khởi đầu của rác
            if any(p.search(line) for p in garbage_patterns):
                out_lines.append("-- [PRUNED] " + line)
                base_indent = len(line) - len(line.lstrip())
                skip_mode = True
                continue
            
            # 2. Xử lý các dòng con (body) của rác
            if skip_mode:
                if not line.strip():
                    out_lines.append("-- [PRUNED] " + line)
                    continue
                
                curr_indent = len(line) - len(line.lstrip())
                # Trong Lean, body của 'have' BẮT BUỘC phải thụt lề sâu hơn base_indent
                if curr_indent > base_indent:
                    out_lines.append("-- [PRUNED] " + line)
                    continue
                else:
                    # Thoát khỏi block rác vì đã gặp một lệnh ngang hàng
                    skip_mode = False
                    
            # 3. Code xịn thì cho qua
            out_lines.append(line)

        return "\n".join(out_lines)
