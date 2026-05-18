from __future__ import annotations

import argparse
import difflib
import json
import re
import textwrap
from pathlib import Path

from gammazero.policy.output_parser import get_lean_code
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.utils.lean_parse import extract_proof_body


LEAN_BLOCK_RE = re.compile(r"```lean4\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)


def final_lean_block(text: str) -> str:
    matches = LEAN_BLOCK_RE.findall(text)
    return matches[-1] if matches else ""


def split_theorem_at_body(code: str) -> tuple[str, str]:
    marker = re.search(r"(?is):=\s*by", code)
    if not marker:
        raise ValueError("Lean block has no `:= by` theorem body")
    return code[: marker.end()], textwrap.dedent(code[marker.end() :]).strip("\n")


def unique_sorry_index(skeleton_body: str) -> int:
    matches = list(re.finditer(r"\bsorry\b", skeleton_body))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one target sorry in prompt scaffold, got {len(matches)}")
    return 0


def build_parent_from_prompt(prompt_code: str, target_body: str) -> str:
    header, prompt_body = split_theorem_at_body(prompt_code)
    target_idx = unique_sorry_index(prompt_body)
    stitched_body = ProofStitcher.stitch(prompt_body, [target_body])
    return f"{header}\n{textwrap.indent(stitched_body, '  ')}"


def direct_parent_score(original_parent: str, patched_parent: str) -> tuple[float, list[str], list[str]]:
    reward = RewardCalculator()
    score = reward.r_env(original_parent, patched_parent, {"pass": True, "warnings": []})
    return (
        score,
        reward._get_clean_proof_lines(original_parent),
        reward._get_clean_proof_lines(patched_parent),
    )


def diff_lines(orig_lines: list[str], patch_lines: list[str]) -> str:
    return "\n".join(
        difflib.unified_diff(
            orig_lines,
            patch_lines,
            fromfile="original_parent",
            tofile="patched_parent",
            lineterm="",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe direct parent-scaffold scoring for a real subgoal tactic output. "
            "This does not modify rollout code."
        )
    )
    parser.add_argument(
        "--json",
        default="outputs/rollouts/gemini3flash/aime_1983_p1.json",
        help="rollout JSON containing subgoal tactic prompts",
    )
    parser.add_argument("--action-id", default="action_17")
    parser.add_argument("--all", action="store_true", help="scan every subgoal tactic prompt")
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())

    nodes = (
        [
            n
            for n in data["nodes"]
            if n.get("type") == "AND"
            and n.get("action_type") == "tactic"
            and "Solve the unique `sorry` placeholder only" in n.get("prompt", "")
        ]
        if args.all
        else [next(n for n in data["nodes"] if n["id"] == args.action_id)]
    )

    rows: list[tuple[str, str, float | None, float, int, int]] = []
    for node in nodes:
        prompt_code = final_lean_block(node["prompt"])
        if not prompt_code:
            continue

        original_target = textwrap.dedent(node["extracted_lean_code"]).strip("\n")

        # This is a stand-in for a successful sorrifier patch. It keeps the exact
        # same parent scaffold from the real prompt and changes only the target
        # subgoal replacement. Using `by sorry` is intentional: it stress-tests
        # whether unchanged sibling/scaffold lines dominate the survival score.
        patched_target = "by\n  sorry"

        try:
            original_parent = build_parent_from_prompt(prompt_code, original_target)
            patched_parent = build_parent_from_prompt(prompt_code, patched_target)
        except ValueError:
            continue
        score, orig_lines, patch_lines = direct_parent_score(original_parent, patched_parent)
        rows.append(
            (
                node["id"],
                node.get("status", ""),
                node.get("metrics", {}).get("r_env"),
                score,
                len(orig_lines),
                len(patch_lines),
            )
        )

    if args.all:
        print("action_id,status,stored_r_env,direct_parent_r_env,orig_lines,patched_lines")
        for row in rows:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]:.6f},{row[4]},{row[5]}")
        return

    node = nodes[0]
    prompt_code = final_lean_block(node["prompt"])
    if not prompt_code:
        raise ValueError(f"{args.action_id} prompt has no lean4 block")

    original_target = textwrap.dedent(node["extracted_lean_code"]).strip("\n")

    # This is a stand-in for a successful sorrifier patch. It keeps the exact
    # same parent scaffold from the real prompt and changes only the target
    # subgoal replacement. Using `by sorry` is intentional: it stress-tests
    # whether unchanged sibling/scaffold lines dominate the survival score.
    patched_target = "by\n  sorry"

    original_parent = build_parent_from_prompt(prompt_code, original_target)
    patched_parent = build_parent_from_prompt(prompt_code, patched_target)
    score, orig_lines, patch_lines = direct_parent_score(original_parent, patched_parent)

    print(f"action_id: {args.action_id}")
    print(f"status: {node.get('status')}")
    print(f"stored_r_env: {node.get('metrics', {}).get('r_env')}")
    print(f"direct_parent_r_env: {score:.6f}")
    print(f"original_parent_lines: {len(orig_lines)}")
    print(f"patched_parent_lines: {len(patch_lines)}")
    print()
    print("target_body_from_real_output:")
    print(original_target)
    print()
    print("direct_parent_diff:")
    print(diff_lines(orig_lines, patch_lines))


if __name__ == "__main__":
    main()
