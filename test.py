import argparse
import difflib
import json
import re
from pathlib import Path

from betazero.core.nodes import ProofState
from betazero.env.ast_parser import get_lean_ast
from betazero.env.lean_env import Lean4ServerScheduler, LeanEnv
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.sorrifier.sorrifier import Sorrifier
from betazero.utils.lean_cmd import build_theorem
from betazero.utils.lean_parse import extract_proof_body


DEFAULT_JSON = "outputs/rollouts/gemini3flash/miniF2F-valid-50/amc12_2000_p12.json"
DEFAULT_ACTION = "action_7"


def strip_comments(code: str) -> str:
    return re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")


def has_real_sorry(code: str) -> bool:
    return bool(re.search(r"\bsorry\b", strip_comments(code)))


def first_lines(text: str, n: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i + 1:>4} | {line}" for i, line in enumerate(lines[:n]))


def build_full_code_from_json(data: dict, action_id: str) -> tuple[str, dict, dict]:
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    action = nodes[action_id]

    parent = None
    for edge in data.get("edges", []):
        if edge.get("relation") == "expanded_to" and edge.get("target") == action_id:
            parent = nodes[edge["source"]]
            break
    if parent is None:
        raise ValueError(f"Cannot find parent state for {action_id}")

    prompt = action.get("prompt", "")
    header = "open BigOperators Nat Real Topology\n"
    match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?)\ntheorem", prompt, re.DOTALL)
    if match:
        header = match.group(1).strip() + "\n"

    full_match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?:= by\n)  sorry\n```", prompt, re.DOTALL)
    proof_body = action.get("extracted_lean_code") or ""
    if full_match:
        return full_match.group(1) + proof_body, action, parent

    state = ProofState(
        goal=parent["content"]["goal"],
        context=parent["content"].get("context", ""),
        header=header,
    )
    return build_theorem(state, proof_body), action, parent


def summarize_verify(label: str, result: dict) -> None:
    print(f"\n=== {label} VERIFY ===")
    print("pass:", result.get("pass"), "complete:", result.get("complete"))
    print("sorries:", len(result.get("sorries", []) or []), "system_errors:", result.get("system_errors") or "")

    errors = result.get("errors", []) or []
    warnings = result.get("warnings", []) or []
    print("errors:", len(errors), "warnings:", len(warnings))

    for msg in errors[:8]:
        pos = msg.get("pos", {})
        data = (msg.get("data") or "").replace("\n", " ")
        print(f"  error L{pos.get('line')}:{pos.get('column')} {data[:240]}")
    if len(errors) > 8:
        print(f"  ... {len(errors) - 8} more errors")

    for msg in warnings[:5]:
        pos = msg.get("pos", {})
        data = (msg.get("data") or "").replace("\n", " ")
        print(f"  warning L{pos.get('line')}:{pos.get('column')} {data[:200]}")
    if len(warnings) > 5:
        print(f"  ... {len(warnings) - 5} more warnings")


def ast_stats(label: str, code: str, verify_result: dict | None = None) -> dict:
    ast = get_lean_ast(code)
    tactic_kinds = [
        n.get("kind", "")
        for n in ast
        if (n.get("kind") or "").startswith("Lean.Parser.Tactic.")
    ]
    sorry_kinds = [k for k in tactic_kinds if "sorry" in k.lower()]

    print(f"\n=== {label} AST ===")
    print("ast_nodes:", len(ast))
    print("raw_tactic_nodes:", len(tactic_kinds), "raw_sorry_tactic_nodes:", len(sorry_kinds))
    print("has_real_sorry:", has_real_sorry(code))
    print("first tactic kinds:")
    for kind in tactic_kinds[:20]:
        print(" ", kind)
    if len(tactic_kinds) > 20:
        print(f"  ... {len(tactic_kinds) - 20} more tactic nodes")

    return {
        "ast": ast,
        "raw_tactic_nodes": len(tactic_kinds),
        "raw_sorry_tactic_nodes": len(sorry_kinds),
    }


def reward_breakdown(original_code: str, patched_code: str, verify_result: dict) -> dict:
    reward = RewardCalculator()
    orig_lines = reward._get_clean_proof_lines(original_code)
    patch_lines = reward._get_clean_proof_lines(patched_code)
    matcher = difflib.SequenceMatcher(None, orig_lines, patch_lines)
    matching_blocks = matcher.get_matching_blocks()
    surviving_lines = sum(match.size for match in matching_blocks)

    dead_warnings = [
        w for w in verify_result.get("warnings", []) or []
        if "unused" in (w.get("data") or "").lower()
        or "does nothing" in (w.get("data") or "").lower()
    ]
    valid_survivors = max(0, surviving_lines - len(dead_warnings))
    base_score = valid_survivors / len(orig_lines) if orig_lines else 0.0

    return {
        "orig_lines": orig_lines,
        "patch_lines": patch_lines,
        "matching_blocks": matching_blocks,
        "surviving_lines": surviving_lines,
        "dead_count": len(dead_warnings),
        "valid_survivors": valid_survivors,
        "base_score": base_score,
        "final_score": base_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--action", default=DEFAULT_ACTION)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log-path", default=None, help="Optional sorrifier trace log path")
    args = parser.parse_args()

    path = Path(args.json)
    data = json.loads(path.read_text(encoding="utf-8"))
    original_code, action, parent = build_full_code_from_json(data, args.action)

    print("JSON:", path)
    print("action:", args.action)
    print("stored status:", action.get("status"))
    print("stored metrics:", action.get("metrics"))
    print("parent goal:", parent.get("content", {}).get("goal"))
    print("\n=== ORIGINAL CODE HEAD ===")
    print(first_lines(original_code, 35))

    scheduler = Lean4ServerScheduler(timeout=args.timeout)
    lean = LeanEnv(scheduler)
    reward = RewardCalculator()
    sorrifier = Sorrifier(scheduler, log_path=args.log_path)

    try:
        before_vr = lean.verify(original_code)
        summarize_verify("BEFORE", before_vr)
        before_ast = ast_stats("BEFORE", original_code, before_vr)

        patched_code = sorrifier.fix_code(original_code)
        patched_body = extract_proof_body(patched_code)
        after_vr = lean.verify(patched_code)
        summarize_verify("AFTER SORRIFIER", after_vr)
        after_ast = ast_stats("AFTER SORRIFIER", patched_code, after_vr)

        r_env = reward.r_env(original_code, patched_code, after_vr)
        breakdown = reward_breakdown(original_code, patched_code, after_vr)

        print("\n=== REWARD ===")
        print("r_env(original, patched, after_vr):", r_env)
        print("orig_clean_lines:", len(breakdown["orig_lines"]))
        print("patch_clean_lines:", len(breakdown["patch_lines"]))
        print("surviving_lcs_lines:", breakdown["surviving_lines"])
        print("dead_warning_count:", breakdown["dead_count"])
        print("valid_survivors:", breakdown["valid_survivors"])
        print("base_score:", breakdown["base_score"])
        print("final_score:", breakdown["final_score"])
        print("matching_blocks:", [
            (m.a, m.b, m.size) for m in breakdown["matching_blocks"] if m.size
        ])

        print("\n=== PATCHED CODE HEAD ===")
        print(first_lines(patched_code, 60))

        print("\n=== PATCHED BODY ===")
        print(patched_body)
    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
