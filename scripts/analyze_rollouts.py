#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


Q_TOLERANCE = 1e-6
HIGH_REWARD_THRESHOLD = 0.75
HIGH_Q_THRESHOLD = 2.0

DIAGNOSTIC_EXPLANATIONS = {
    "cycle_detected": "Graph contains a directed cycle. This usually means decomposition generated a repeated state/goal path; extraction may avoid infinite recursion, but the search has spent budget revisiting itself.",
    "failed_full_r_env": "Valid but incomplete tactic: `r_env` reached 1.0 on a `FAILED` action. The code is syntactically/logically sound (survival), but didn't close the goal. This represents a search inefficiency where the agent 'stalls' at high-reward partial progress.",
    "high_q_suspicious_skeleton": "A skeleton has high backed-up `Q_value` despite weak dependency evidence or child proofs still containing `sorry`. This is the strongest signal that backup is amplifying a structurally suspicious decomposition.",
    "missing_extracted_code": "An action has no parsed Lean code. The raw model output may be malformed, outside the expected code block format, or parser-incompatible.",
    "open_root_high_avg_reward": "The root remains `OPEN` while average action `r_env` is high. This means local rewards are optimistic but do not translate into a complete proof.",
    "q_backup_mismatch": "Observed `Q_value` does not match the expected deterministic backup formula from current node status/rewards. This usually means stale metrics after mutation or a different backup rule was used.",
    "solved_skeleton_with_unsolved_children": "A skeleton is marked `SOLVED` while at least one child state is not solved. This points to an override bug or garbage-pruning path that accepted an incomplete stitched proof.",
    "solved_state_proof_contains_sorry": "An OR state is marked `SOLVED` but its stored `proof_body` still contains a real `sorry`. Any final proof extracted through this state is not kernel-complete.",
    "solved_tactic_contains_sorry": "A tactic action is marked `SOLVED` but the extracted Lean code still contains a real `sorry`. The verifier/status and stored code disagree.",
}


def strip_lean_comments(code: str) -> str:
    return re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")


def has_real_sorry(code: str) -> bool:
    return bool(re.search(r"\bsorry\b", strip_lean_comments(code or "")))


def preview(text: Any, limit: int = 120) -> str:
    s = "" if text is None else str(text).replace("\n", "\\n")
    return s[:limit] + ("..." if len(s) > limit else "")


def stable_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".json" else []
    return sorted(path.rglob("*.json"))


def load_rollout_file(path: Path, base_dir: Path | None = None) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    theorem = path.name if base_dir is None else str(path.relative_to(base_dir))
    return analyze_rollout_data(data, theorem=theorem, source_path=str(path))


def _relationships(edges: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    state_to_actions: dict[str, list[str]] = defaultdict(list)
    action_to_children: dict[str, list[str]] = defaultdict(list)
    action_to_parent: dict[str, str] = {}
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        rel = edge.get("relation")
        if not src or not tgt:
            continue
        if rel == "expanded_to":
            state_to_actions[src].append(tgt)
            action_to_parent[tgt] = src
        elif rel == "subgoal":
            action_to_children[src].append(tgt)
    return state_to_actions, action_to_children, action_to_parent





def detect_cycles(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    adj = defaultdict(list)
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src and tgt:
            adj[src].append(tgt)
    
    cycles = []
    visited = set()
    stack = []
    stack_set = set()

    def find_cycles(u):
        if len(cycles) >= limit: return
        visited.add(u)
        stack.append(u)
        stack_set.add(u)
        for v in adj.get(u, []):
            if v in stack_set:
                idx = stack.index(v)
                cyc = stack[idx:]
                repeated_goals = []
                seen_goals = {}
                for node_id in cyc:
                    node = nodes.get(node_id, {})
                    goal = node.get("content", {}).get("goal") if node.get("type") == "OR" else None
                    if goal:
                        if goal in seen_goals: repeated_goals.append(goal)
                        seen_goals[goal] = node_id
                
                cycles.append({
                    "cycle_length": len(cyc),
                    "cycle_nodes": " -> ".join(cyc),
                    "repeated_goals": " | ".join(preview(g, 160) for g in repeated_goals),
                    "cycle_edges": " ; ".join(f"edge_placeholder" for _ in cyc),
                })
            elif v not in visited:
                find_cycles(v)
        stack.pop()
        stack_set.remove(u)

    for node in list(nodes.keys()):
        if node not in visited:
            find_cycles(node)
    return cycles


def _shortest_solved_path(
    root_id: str,
    nodes: dict[str, dict[str, Any]],
    state_to_actions: dict[str, list[str]],
    action_to_children: dict[str, list[str]],
) -> list[str]:
    """Pick a compact solved proof route, preferring solved actions with higher Q."""
    if root_id not in nodes:
        return []
    path: list[str] = []
    visiting: set[str] = set()

    def visit_state(state_id: str) -> None:
        if state_id in visiting:
            path.append(f"{state_id}(cycle)")
            return
        visiting.add(state_id)
        path.append(state_id)
        actions = [
            a for a in state_to_actions.get(state_id, [])
            if nodes.get(a, {}).get("status") == "SOLVED"
        ]
        actions.sort(key=lambda a: nodes[a].get("metrics", {}).get("Q_value", 0.0), reverse=True)
        if not actions:
            visiting.remove(state_id)
            return
        action_id = actions[0]
        path.append(action_id)
        for child in action_to_children.get(action_id, []):
            visit_state(child)
        visiting.remove(state_id)

    visit_state(root_id)
    return path


def analyze_rollout_data(data: dict[str, Any], *, theorem: str, source_path: str = "") -> dict[str, Any]:
    raw_nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    nodes = {n.get("id"): dict(n) for n in raw_nodes if n.get("id")}
    state_to_actions, action_to_children, action_to_parent = _relationships(edges)

    for action_id, parent_id in action_to_parent.items():
        parent = nodes.get(parent_id, {})
        nodes.get(action_id, {})["_parent_depth"] = parent.get("depth", 0)

    root_id = data.get("root_id", "state_0")
    root_node = nodes.get(root_id, {})
    root_status = root_node.get("status", "UNKNOWN")
    theorem_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []

    indegree = Counter()
    outdegree = Counter()
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src:
            outdegree[src] += 1
        if tgt:
            indegree[tgt] += 1
        edge_rows.append(
            {
                "theorem": theorem,
                "source_path": source_path,
                "source": src,
                "target": tgt,
                "relation": edge.get("relation"),
            }
        )

    q_mismatches = 0
    action_q_expected: dict[str, float] = {}
    for node_id, node in nodes.items():
        if node.get("type") != "AND":
            continue
        metrics = node.get("metrics", {})
        r_env = float(metrics.get("r_env", 0.0) or 0.0)
        solved = 1.0 if node.get("status") == "SOLVED" else 0.0
        if node.get("action_type") == "tactic":
            expected_q = r_env + solved
        else:
            child_vs = [
                float(nodes.get(c, {}).get("metrics", {}).get("V_value", 0.0) or 0.0)
                for c in action_to_children.get(node_id, [])
            ]
            future = min(child_vs) if child_vs else 0.0
            r_dep = float(metrics.get("r_dep", 0.0) or 0.0)
            expected_q = r_env + solved * (r_dep + future)
        action_q_expected[node_id] = expected_q
        observed_q = metrics.get("Q_value")
        if observed_q is not None and abs(float(observed_q) - expected_q) > Q_TOLERANCE:
            q_mismatches += 1
            anomaly_rows.append(
                _anomaly(theorem, source_path, node_id, "q_backup_mismatch",
                         f"observed={observed_q}, expected={expected_q:.6g}")
            )

    for node_id, node in nodes.items():
        ntype = node.get("type")
        metrics = node.get("metrics", {})
        content = node.get("content", {})
        is_action = ntype == "AND"
        is_state = ntype == "OR"
        extracted = node.get("extracted_lean_code") or ""
        proof_body = content.get("proof_body", "") if isinstance(content, dict) else ""
        prompt = node.get("prompt") or ""
        parent_id = action_to_parent.get(node_id, "")
        child_ids = action_to_children.get(node_id, [])
        action_type = node.get("action_type", "")
        depth = node.get("depth")
        if depth is None and is_action:
            depth = nodes.get(parent_id, {}).get("depth", 0) + 0.5

        row = {
            "theorem": theorem,
            "source_path": source_path,
            "id": node_id,
            "type": ntype,
            "status": node.get("status"),
            "depth": depth,
            "parent": parent_id,
            "children_count": len(child_ids),
            "in_degree": indegree[node_id],
            "out_degree": outdegree[node_id],
            "action_type": action_type,
            "synthetic_patch": bool(str(prompt).startswith("[SYNTHETIC_PATCH]")),
            "goal": content.get("goal", "") if isinstance(content, dict) else "",
            "goal_hash": stable_hash(content.get("goal", "")) if isinstance(content, dict) else "",
            "goal_preview": preview(content.get("goal", "")) if isinstance(content, dict) else "",
            "context_len": len(content.get("context", "")) if isinstance(content, dict) else 0,
            "content_len": len(str(node.get("content", ""))),
            "code_len": len(extracted),
            "proof_len": len(proof_body),
            "has_sorry": has_real_sorry(extracted if is_action else proof_body),
            "r_env": metrics.get("r_env"),
            "r_dep": metrics.get("r_dep"),
            "Q_value": metrics.get("Q_value"),
            "V_value": metrics.get("V_value"),
            "expected_Q": action_q_expected.get(node_id),
        }
        node_rows.append(row)

        child_proofs_have_sorry = False
        if is_action and action_type == "skeleton":
            for child_id in child_ids:
                child = nodes.get(child_id, {})
                child_proof = child.get("content", {}).get("proof_body", "")
                if has_real_sorry(child_proof):
                    child_proofs_have_sorry = True
                    break
        unsolved_children = [
            child_id for child_id in child_ids
            if nodes.get(child_id, {}).get("status") != "SOLVED"
        ]

        r_env = metrics.get("r_env")
        r_dep = metrics.get("r_dep")
        q_value = metrics.get("Q_value")
        status = node.get("status")
        if is_action and status == "FAILED" and r_env == 1.0:
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "failed_full_r_env", "FAILED action has r_env=1.0"))
        if is_action and action_type == "tactic" and status == "SOLVED" and has_real_sorry(extracted):
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "solved_tactic_contains_sorry", "SOLVED tactic code contains real sorry"))
        if is_state and status == "SOLVED" and has_real_sorry(proof_body):
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "solved_state_proof_contains_sorry", "SOLVED state proof body contains real sorry"))
        if is_action and action_type == "skeleton" and status == "SOLVED" and unsolved_children:
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "solved_skeleton_with_unsolved_children", f"SOLVED skeleton has unsolved children: {', '.join(unsolved_children[:20])}"))
        if is_action and action_type == "skeleton" and q_value is not None and q_value >= HIGH_Q_THRESHOLD and (r_dep == 0 or child_proofs_have_sorry):
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "high_q_suspicious_skeleton", f"skeleton Q={q_value}, r_dep={r_dep}, child_proofs_have_sorry={child_proofs_have_sorry}"))
        if is_action and not extracted.strip():
            anomaly_rows.append(_anomaly(theorem, source_path, node_id, "missing_extracted_code", "action has empty extracted_lean_code"))

    avg_action_reward = np.mean([r["r_env"] for r in node_rows if r["type"] == "AND" and r["r_env"] is not None]) if node_rows else 0.0
    if root_status == "OPEN" and avg_action_reward >= HIGH_REWARD_THRESHOLD:
        anomaly_rows.append(_anomaly(theorem, source_path, root_id, "open_root_high_avg_reward", f"root OPEN but avg action r_env={avg_action_reward:.3f}"))

    cycle_rows = detect_cycles(nodes, edges)
    for cyc in cycle_rows:
        anomaly_rows.append(
            _anomaly(theorem, source_path, cyc["cycle_nodes"], "cycle_detected",
                     f"length={cyc['cycle_length']}; goals={cyc['repeated_goals']}; edges={cyc['cycle_edges']}")
        )

    action_rows = [r for r in node_rows if r["type"] == "AND"]
    state_rows = [r for r in node_rows if r["type"] == "OR"]
    depths = [r["depth"] for r in node_rows if r["depth"] is not None]
    depth_width = Counter(int(math.floor(float(d))) for d in depths)
    or_shared = sum(1 for r in state_rows if r["in_degree"] > 1)
    solved_path = _shortest_solved_path(root_id, nodes, state_to_actions, action_to_children)

    theorem_rows.append(
        {
            "theorem": theorem,
            "source_path": source_path,
            "root_id": root_id,
            "root_status": root_status,
            "solved": root_status == "SOLVED",
            "total_nodes": len(raw_nodes),
            "total_edges": len(edges),
            "or_nodes": len(state_rows),
            "and_nodes": len(action_rows),
            "tactic_actions": sum(1 for r in action_rows if r["action_type"] == "tactic"),
            "skeleton_actions": sum(1 for r in action_rows if r["action_type"] == "skeleton"),
            "synthetic_skeletons": sum(1 for r in action_rows if r["action_type"] == "skeleton" and r["synthetic_patch"]),
            "solved_actions": sum(1 for r in action_rows if r["status"] == "SOLVED"),
            "failed_actions": sum(1 for r in action_rows if r["status"] == "FAILED"),
            "open_actions": sum(1 for r in action_rows if r["status"] == "OPEN"),
            "max_depth": max(depth_width.keys(), default=0),
            "max_width": max(depth_width.values(), default=0),
            "shared_or_states": or_shared,
            "cycles": len(cycle_rows),
            "q_mismatches": q_mismatches,
            "avg_r_env": float(np.mean([r["r_env"] for r in action_rows if r["r_env"] is not None])) if action_rows else 0.0,
            "avg_r_dep": float(np.mean([r["r_dep"] for r in action_rows if r["r_dep"] is not None])) if action_rows else 0.0,
            "avg_q": float(np.mean([r["Q_value"] for r in action_rows if r["Q_value"] is not None])) if action_rows else 0.0,
            "solved_path_len": len(solved_path),
            "solved_path": " -> ".join(solved_path),
            "anomalies": len(anomaly_rows),
        }
    )

    return {
        "theorem_rows": theorem_rows,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "anomaly_rows": anomaly_rows,
        "nodes": nodes,
        "edges": edges,
        "data": data,
    }


def _anomaly(theorem: str, source_path: str, node_id: str, kind: str, detail: str) -> dict[str, Any]:
    return {
        "theorem": theorem,
        "source_path": source_path,
        "node_id": node_id,
        "kind": kind,
        "detail": detail,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dot_escape(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_summary(path: Path, theorem_df: pd.DataFrame, node_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action_df = node_df[node_df["type"] == "AND"] if not node_df.empty else pd.DataFrame()
    state_df = node_df[node_df["type"] == "OR"] if not node_df.empty else pd.DataFrame()
    solved = int(theorem_df["solved"].sum()) if not theorem_df.empty else 0
    total = len(theorem_df)
    pass_rate = solved / total if total else 0.0
    lines = [
        "# Rollout Analysis Report",
        "",
        "## Design Intent",
        "",
        "The rollout JSON encodes an asymmetric AND/OR proof graph. OR nodes are Lean proof states; AND nodes are generated actions. Tactics are terminal actions, while skeletons create subgoal states. `r_env` is the immediate syntactic/repair survival reward, `r_dep` is dependency-aware skeleton quality, and `Q/V` are deterministic AND/OR backups.",
        "",
        "## Dataset Health",
        "",
        f"- Theorems: {total}",
        f"- Solved roots: {solved} ({pass_rate:.1%})",
        f"- Nodes: {len(node_df)} total, {len(state_df)} OR, {len(action_df)} AND",
        f"- Edges: {int(theorem_df['total_edges'].sum()) if not theorem_df.empty else 0}",
        f"- Avg nodes/theorem: {theorem_df['total_nodes'].mean():.2f}" if total else "- Avg nodes/theorem: 0",
        f"- Avg max depth: {theorem_df['max_depth'].mean():.2f}" if total else "- Avg max depth: 0",
        f"- Shared OR states: {int(theorem_df['shared_or_states'].sum()) if total else 0}",
        f"- Cycles detected: {int(theorem_df['cycles'].sum()) if total else 0}",
        "",
        "## Reward Behavior",
        "",
    ]
    if not action_df.empty:
        for col in ["r_env", "r_dep", "Q_value"]:
            vals = pd.to_numeric(action_df[col], errors="coerce").dropna()
            lines.append(f"- {col}: mean={vals.mean():.4f}, median={vals.median():.4f}, zeros={(vals == 0).sum()}, ones={(vals == 1).sum()}")
        full_failed = len(anomaly_df[anomaly_df["kind"] == "failed_full_r_env"]) if not anomaly_df.empty else 0
        lines.append(f"- FAILED actions with full `r_env`: {full_failed}")
        q_mismatches = int(theorem_df["q_mismatches"].sum()) if total else 0
        lines.append(f"- Q backup mismatches: {q_mismatches}")
    else:
        lines.append("- No action rows found.")
    lines += [
        "",
        "## Graph Behavior",
        "",
    ]
    if not theorem_df.empty:
        largest = theorem_df.sort_values("total_nodes", ascending=False).head(5)
        lines.append("- Largest graphs:")
        for _, row in largest.iterrows():
            lines.append(f"  - `{row['theorem']}`: nodes={row['total_nodes']}, max_depth={row['max_depth']}, status={row['root_status']}")
    lines += [
        "",
        "## Diagnostic & Efficiency Analysis",
        "",
    ]
    if anomaly_df.empty:
        lines.append("- No diagnostic issues detected.")
    else:
        counts = anomaly_df["kind"].value_counts()
        for kind, count in counts.head(15).items():
            explanation = DIAGNOSTIC_EXPLANATIONS.get(
                kind,
                "No built-in explanation is registered for this diagnostic kind yet.",
            )
            lines.append(f"- `{kind}` ({count}): {explanation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")





def maybe_refresh_rollouts(files: list[Path], base_dir: Path, out_dir: Path, mode: str, suspect_theorems: set[str]) -> list[Path]:
    if mode == "none":
        return files
    selected = []
    for path in files:
        rel = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else path.name
        if mode == "all" or rel in suspect_theorems or path.name in suspect_theorems:
            selected.append(path)
    if not selected:
        return files

    refreshed_dir = out_dir / "refreshed_json"
    refreshed_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in selected:
        rel = path.relative_to(base_dir) if path.is_relative_to(base_dir) else Path(path.name)
        dst = refreshed_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(dst)

    print(f"[refresh-lean] Refreshing {len(copied)} copied JSON files under {refreshed_dir}", file=sys.stderr)
    try:
        from recalculate_rollout_rdep import process_file
        from betazero.env.lean_env import LeanEnv
        from betazero.env.lean_verifier import Lean4ServerScheduler
        from betazero.search.reward.calculator import RewardCalculator
        from betazero.search.reward.reward_assigner import DependencyRewardAssigner
    except Exception as exc:
        print(f"[refresh-lean] Could not import Lean refresh pipeline: {exc}", file=sys.stderr)
        return files

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=120, name="analysis_refresh")
    try:
        lean = LeanEnv(scheduler)
        reward_calc = RewardCalculator()
        assigner = DependencyRewardAssigner(lean, reward_calc)
        for path in copied:
            process_file(path, lean, reward_calc, assigner)
    finally:
        scheduler.close()

    refreshed_by_rel = {
        (p.relative_to(refreshed_dir) if p.is_relative_to(refreshed_dir) else Path(p.name)): p
        for p in copied
    }
    result = []
    for path in files:
        rel = path.relative_to(base_dir) if path.is_relative_to(base_dir) else Path(path.name)
        result.append(refreshed_by_rel.get(rel, path))
    return result


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    files = json_files(in_path)
    if not files:
        raise SystemExit(f"No JSON files found at {in_path}")
    base_dir = in_path if in_path.is_dir() else in_path.parent

    first_pass = [load_rollout_file(path, base_dir=base_dir) for path in files]
    first_theorem_df = pd.DataFrame([r for a in first_pass for r in a["theorem_rows"]])
    first_anomaly_df = pd.DataFrame([r for a in first_pass for r in a["anomaly_rows"]])
    suspect_theorems = set(first_anomaly_df["theorem"].unique()) if not first_anomaly_df.empty else set()
    files = maybe_refresh_rollouts(files, base_dir, out_dir, args.refresh_lean, suspect_theorems)

    analyses = [load_rollout_file(path, base_dir=base_dir if path.is_relative_to(base_dir) else path.parent) for path in files]
    theorem_rows = [r for a in analyses for r in a["theorem_rows"]]
    node_rows = [r for a in analyses for r in a["node_rows"]]
    edge_rows = [r for a in analyses for r in a["edge_rows"]]
    anomaly_rows = [r for a in analyses for r in a["anomaly_rows"]]

    theorem_df = pd.DataFrame(theorem_rows)
    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)
    anomaly_df = pd.DataFrame(anomaly_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "theorems.csv", theorem_rows)
    write_csv(out_dir / "nodes.csv", node_rows)
    write_csv(out_dir / "edges.csv", edge_rows)
    write_csv(out_dir / "anomalies.csv", anomaly_rows)
    write_summary(out_dir / "summary.md", theorem_df, node_df, anomaly_df)


    print(f"Wrote analysis for {len(files)} rollout JSON files to {out_dir}")
    print(f"Summary: {out_dir / 'summary.md'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BetaZero rollout AND/OR graph JSON files.")
    parser.add_argument("--json-dir", required=True, help="Rollout JSON file or directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV/plots/reports.")
    parser.add_argument("--refresh-lean", choices=["none", "suspect", "all"], default="none")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
