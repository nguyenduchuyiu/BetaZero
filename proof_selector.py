"""
Score, rank, and select the best LLM proof candidates.

Pipeline (skip initial verify):
  1. Sorrify all candidates (SyntaxCorrector -> AutoSorrifier).
  2. Verify sorrified results; keep only those that pass.
  3. Extract subgoals from passing candidates.
"""
import gc
import os
import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from prover.lean.verifier import verify_lean4_file, Lean4ServerScheduler
from utils.syntax_repair import SyntaxCorrector
from utils.auto_sorrifier import AutoSorrifier, PersistentASTDaemon, REPL_DIR
from utils.proof_state_extractor import extract_queries


# ===========================================================================
# 1. ERROR CLASSIFICATION
# ===========================================================================

# Pre-compile regex patterns for performance
ERROR_PATTERNS = {
    "semantic": re.compile(
        r"type mismatch|failed to synthesize|linarith failed|omega failed|"
        r"nlinarith failed|ring_nf failed|simp made no progress|unsolved goals|"
        r"tactic .+ failed|application type mismatch|has type.*but is expected to have type|"
        r"function expected|no goals to be solved|mod_cast|norm_num failed|norm_cast failed", 
        re.IGNORECASE
    ),
    "missing_constant": re.compile(
        r"unknown constant|unknown identifier|unknown namespace|not found", 
        re.IGNORECASE
    ),
    "syntax": re.compile(
        r"unexpected token|expected command|expected '.+'|unterminated|"
        r"unexpected end of input|expected token|unknown tactic|invalid escape sequence", 
        re.IGNORECASE
    )
}

CATEGORY_WEIGHT = {
    "semantic": 0,          # Best: survived parsing and logical checks
    "missing_constant": 1,
    "other": 2,
    "syntax": 3,            # Worst: code is malformed
}

def classify_error(error_msg: str) -> str:
    """Classifies a Lean 4 error message into predefined categories."""
    for category, pattern in ERROR_PATTERNS.items():
        if pattern.search(error_msg):
            return category
    return "other"

def classify_errors(errors: List[Dict]) -> Dict[str, List[Dict]]:
    """Groups a list of REPL errors by their category."""
    groups = {"semantic": [], "missing_constant": [], "syntax": [], "other": []}
    for e in errors:
        groups[classify_error(e["data"])].append(e)
    return groups

def is_garbage_lean(code_body: str) -> bool:
    """Phát hiện LLM sinh văn xuôi tiếng Anh hoặc LaTeX thay vì code Lean 4."""
    # 1. Phát hiện LaTeX rác (Lean dùng Unicode, không dùng lệnh LaTeX thô)
    latex_patterns = [r"\$.*?\$", r"\\\[.*?\\\]", r"\\mathbb", r"\\to\s", r"\\rightarrow", r"\\frac", r"\\sqrt"]
    
    # 2. Phát hiện văn xuôi tiếng Anh ở đầu dòng
    prose_patterns = [
        r"^[ \t]*We (have|need|can|will|know)", 
        r"^[ \t]*From (our|the|this)", 
        r"^[ \t]*Notice that", 
        r"^[ \t]*Therefore",
        r"^[ \t]*Let's",
        r"^[ \t]*Here,"
    ]
    
    for p in latex_patterns + prose_patterns:
        if re.search(p, code_body, re.MULTILINE | re.IGNORECASE):
            return True
    return False

# ===========================================================================
# 2. DATA STRUCTURES & SCORING
# ===========================================================================

@dataclass
class ScoredProof:
    idx: int
    code: str
    status: str                         # COMPLETE / PASS / FAIL
    verify_time: float
    errors: List[Dict] = field(default_factory=list)
    sorries: List[Dict] = field(default_factory=list)
    error_groups: Dict = field(default_factory=dict)
    
    # Computed during Stage 1
    worst_category: str = "syntax"
    first_error_line: int = 9999
    raw_score: float = 999.0            
    
    # Computed during Stage 2 (Sorrify & Repair)
    sorrified_code: str = ""
    num_sorries: int = 9999
    num_preserved_steps: int = 0
    final_score: float = 999.0          

    def compute_raw_score(self):
        """Scores initial candidate based on verification status and first error line."""
        if self.status == "COMPLETE":
            self.raw_score = -1000.0
            return
        if self.status == "PASS":
            self.raw_score = -500.0
            return
        if not self.errors:
            self.raw_score = 0.0
            return

        # Filter out consequence errors to find the root cause
        real_errors = [e for e in self.errors if not re.search(r"unsolved goals|no goals", e["data"], re.IGNORECASE)]
        real_errors = real_errors or self.errors
        real_errors.sort(key=lambda e: e["pos"]["line"])
        
        first_err = real_errors[0]
        self.first_error_line = first_err["pos"]["line"]
        self.worst_category = classify_error(first_err["data"])

        # Formula: Category Penalty - Depth Bonus - Time Penalty
        cat_score = CATEGORY_WEIGHT.get(self.worst_category, 2) * 1000
        line_bonus = self.first_error_line * 10
        self.raw_score = cat_score - line_bonus - self.verify_time

    def compute_final_score(self):
        """Scores structural quality: more preserved steps = better, fewer sorries = tiebreaker."""
        if self.status == "COMPLETE":
            self.final_score = -1000.0
        elif not self.sorrified_code:
            self.final_score = self.raw_score
        else:
            self.final_score = -(self.num_preserved_steps * 100) + (self.num_sorries * 50)


# ===========================================================================
# 3. PIPELINE STAGES
# ===========================================================================

def _sorrify_one(idx: int, code: str, verifier: Lean4ServerScheduler, ast_server: PersistentASTDaemon) -> ScoredProof:
    """Sorrify one candidate (no prior verify), dùng chung verifier + AST daemon."""
    sp = ScoredProof(
        idx=idx,
        code=code,
        status="PENDING",
        verify_time=0,
        errors=[],
        sorries=[],
        error_groups={},
    )
    header_split = code.split(":= by")
    if len(header_split) > 1:
        body = header_split[-1]
        if is_garbage_lean(body):
            print(f"  [sorrify] #{idx:2d} |SKIPPED: Detected garbage English/LaTeX.")
            sp.sorrified_code = header_split[0] + ":= by\n  sorry\n"
            sp.num_sorries = 1
            sp.num_preserved_steps = 0
            sp.status = "FAIL"
            sp.compute_final_score()
            return sp
    try:
        code_corrected = SyntaxCorrector(code).correct_text()
        checker = AutoSorrifier(code_corrected, ast_server, verifier, max_cycles=50)
        sorrified = checker.fix_code()
        sp.sorrified_code = sorrified
        lines = [l.strip() for l in sorrified.splitlines() if l.strip()]
        sp.num_sorries = lines.count("sorry")
        sp.num_preserved_steps = len(lines) - sp.num_sorries
    except Exception as e:
        print(f"  [sorrify] failed for #{idx}: {e}")
        sp.sorrified_code, sp.num_sorries, sp.num_preserved_steps = "", 9999, 0
    sp.compute_final_score()
    return sp


def _extract_one(sp: ScoredProof, original_target: str = "") -> List[Dict]:
    """Worker function for Stage 3."""
    if not sp.sorrified_code:
        return []
    try:
        results = []
        for q in extract_queries(sp.sorrified_code):
            normalized = re.sub(r"\(([^()]+?)\s*:\s*[^()]+?\)", r"\1", q["goal"])
            normalized = re.sub(r"\s+", " ", normalized).strip()
            
            if original_target and normalized == original_target:
                continue # Skip trivial goals (identical to original target)
                
            results.append({
                "source_idx": sp.idx,
                "line": q["line"],
                "goal": q["goal"],
                "hypotheses": q["hypotheses"],
                "raw": q["raw"],
                "code_prefix": sp.sorrified_code,
            })
        return results
    except Exception as e:
        print(f"  [extract] failed for #{sp.idx}: {e}")
        return []

def select_and_extract(
    header: str,
    proof_bodies: List[str],
    top_k: int = 5,
    verify_timeout: int = 300,
    max_workers: int = 2,
    verifier: "Lean4ServerScheduler | None" = None,
    ast_server: "PersistentASTDaemon | None" = None,
) -> Dict:
    """Orchestrates pipeline: Sorrify all -> Verify -> Extract from passing."""
    
    codes = [header + body for body in proof_bodies]
    for code in codes:
        with open("code.lean", "w") as f:
            f.write(code)
        break
    
    _own_verifier = verifier is None
    _own_ast = ast_server is None

    if _own_verifier:
        print(f"\nInitializing Verifier Pool (max_workers={max_workers}) and AST Daemon...")
        verifier = Lean4ServerScheduler(max_concurrent_requests=max_workers, timeout=verify_timeout, memory_limit=-1, name="proof_selector")
    if _own_ast:
        ast_server = PersistentASTDaemon(REPL_DIR)

    try:
        # --- Stage 1: Sorrify all (no prior verify) ---
        print(f"\n=== Stage 1: Sorrifying {len(codes)} candidates ===")
        sorrified: List[ScoredProof] = []

        # Sử dụng ThreadPool; việc nặng đã nằm trong các process con của Lean
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_sorrify_one, idx, code, verifier, ast_server): idx for idx, code in enumerate(codes)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                    sorrified.append(res)
                    print(f"  [sorrified] #{res.idx:2d} | Sorries: {res.num_sorries:<3} | Preserved: {res.num_preserved_steps}")
                except Exception as e:
                    print(f"  [Worker Error Stage 1] #{idx}: {e}")

        gc.collect()

        # --- Stage 2: Batch verify sorrified; keep only passing ---
        print(f"\n=== Stage 2: Verifying sorrified results (batched via Lean4ServerScheduler) ===")
        verified: List[ScoredProof] = []
        tasks = []
        owners: List[ScoredProof] = []

        for sp in sorrified:
            if not sp.sorrified_code:
                sp.status = "FAIL"
                verified.append(sp)
                continue
            tasks.append(dict(code=sp.sorrified_code, timeout=verify_timeout))
            owners.append(sp)

        if tasks:
            req_ids = verifier.submit_all_request(tasks)
            outputs = verifier.get_all_request_outputs(req_ids)
            for sp, r in zip(owners, outputs):
                sp.status = "COMPLETE" if r.get("complete") else ("PASS" if r.get("pass") else "FAIL")
                sp.verify_time = r.get("verify_time", 0)
                sp.errors = r.get("errors", [])
                sp.sorries = r.get("sorries", [])
                verified.append(sp)
                mark = "OK" if sp.status in ("COMPLETE", "PASS") else "FAIL"
                print(f"  [verify] #{sp.idx:2d} | {sp.status:8s} | {mark}")

        gc.collect()
    
        passing = [sp for sp in verified if sp.status in ("COMPLETE", "PASS")]
        
        if not passing:
            print("\n[!] No candidates passed verification.")
            return {"ranked": verified, "selected": [], "subgoals": []}
        
        passing.sort(key=lambda s: (-s.num_preserved_steps, s.num_sorries))
        selected = passing[:top_k]
        
        # --- Stage 3: Extract Subgoals (sequential, nhẹ) ---
        print(f"\n=== Stage 3: Extracting subgoals from {len(selected)} passing candidates ===")
        all_subgoals: List[Dict] = []
        seen_goals = set()
        
        original_target, lemma_name = "", "unknown"
        if selected:
            decl_match = re.search(r"(?:lemma|theorem)\s+(\w+)", selected[0].code)
            lemma_name = decl_match.group(1) if decl_match else "unknown"
            
            idx = selected[0].code.find(":= by")
            if idx != -1:
                matches = list(re.finditer(r"\)\s*:\s*", selected[0].code[:idx]))
                if matches:
                    target_raw = selected[0].code[:idx][matches[-1].end():].strip()
                    original_target = re.sub(
                        r"\s+",
                        " ",
                        re.sub(r"\(([^()]+?)\s*:\s*[^()]+?\)", r"\1", target_raw),
                    )

        for sp in selected:
            try:
                result_list = _extract_one(sp, original_target)
                for sub_idx, sg in enumerate(result_list, 1):
                    if sg["goal"] not in seen_goals:
                        seen_goals.add(sg["goal"])
                        sg["name"] = f"{lemma_name}_c{sp.idx}_s{sub_idx}"
                        sg["sorry_index"] = sub_idx
                        all_subgoals.append(sg)
                        print(f"  [extracted] {sg['name']} | Goal: {sg['goal'][:60]}...")
            except Exception as e:
                print(f"  [Worker Error Stage 3] #{sp.idx}: {e}")

        gc.collect()
        print(f"\n=== Pipeline Complete. Total unique subgoals: {len(all_subgoals)} ===")
        return {"ranked": verified, "selected": selected, "subgoals": all_subgoals}
    finally:
        if _own_verifier:
            verifier.close()
        if _own_ast:
            ast_server.close()


# ===========================================================================
# 4. CLI ENTRY POINT & PARSER
# ===========================================================================

def parse_log_file(file_path: str) -> Tuple[str, List[str]]:
    """Parses the prompt log to extract the hardcoded header and proof bodies."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = '''import Mathlib
open BigOperators Real Nat Topology Rat
lemma aime_1983_p3_1_1
  (f : ℝ → ℝ)
  (h₀ : ∀ (x : ℝ), f x = x ^ (2 : ℕ) + ((18 : ℝ) * x + (30 : ℝ)) - (2 : ℝ) * √(x ^ (2 : ℕ) + ((18 : ℝ) * x + (45 : ℝ))))
  (h₁ : Fintype (↑(f ⁻¹' {(0 : ℝ)}) : Type)) :
  ∏ x ∈ (f ⁻¹' {(0 : ℝ)}).toFinset, x = (20 : ℝ) := by
'''
    proof_bodies = []
    
    in_parsed_block = False
    current_body = []

    for idx, line in enumerate(lines):
        # Extract Proof Bodies
        stripped = line.strip()
        if stripped == "--- PARSED OUTPUT ---":
            in_parsed_block = True
            current_body = []
            continue
            
        if in_parsed_block:
            if stripped.startswith("--- ") or stripped.startswith("===== "):
                in_parsed_block = False
                content = "".join(current_body).lstrip("\n").rstrip()
                if content and re.search(r"(have |sorry|rw |simp|linarith|omega|calc|intro |apply |exact )", content):
                    proof_bodies.append(content)
            else:
                current_body.append(line)

    return header, proof_bodies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score proof candidates and extract subgoals")
    parser.add_argument("--input", type=str, default="logs.log")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verify-timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16, help="Max parallel verification workers")
    parser.add_argument("--output", type=str, default="output/subgoals.json", help="Save results to JSON file")
    args = parser.parse_args()

    header, proof_bodies = parse_log_file(args.input)

    if not proof_bodies:
        print(f"[!] No proof bodies found in {args.input}.")
        exit(1)

    print(f"[*] Loaded {len(proof_bodies)} proof candidates. Extracted header size: {len(header)} chars.")

    result = select_and_extract(
        header=header,
        proof_bodies=proof_bodies,
        top_k=args.top_k,
        verify_timeout=args.verify_timeout,
        max_workers=args.workers,
    )

    # Save to JSON
    if args.output:
        out_data = {
            "ranked": [
                {
                    "idx": s.idx, "raw_score": s.raw_score, "final_score": s.final_score,
                    "status": s.status, "worst_category": s.worst_category,
                    "first_error_line": s.first_error_line, "num_sorries": s.num_sorries,
                    "num_preserved_steps": s.num_preserved_steps, "verify_time": s.verify_time
                }
                for s in result["ranked"]
            ],
            "subgoals": result["subgoals"],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding='utf-8') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Results successfully saved to {args.output}")