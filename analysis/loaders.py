"""Load GammaZero rollout and Kimina flat_sampling outputs into unified records.

A "problem record" is a dict with:
    name, split, system, solved, has_data, raw   (and system-specific fields)

GammaZero rollouts (system='gammazero'):
    metadata: dict from search_metadata
    nodes:    list of node dicts
    edges:    list of edge dicts
    solved_root_only: bool — solved by a depth-0 tactic
    solved_via_skeleton: bool — solved but used depth>0
    n_nodes, n_or, n_and, n_skeleton_and, n_tactic_and
    used_total, lean_verify_calls, patch_verify_calls, max_depth_reached
    skeleton_pipeline: dict
    skeleton_commitment: dict

Kimina flat_sampling (system='kimina'):
    solved: bool — assembled_main_theorem.lean has no 'sorry' literal
    n_recursion_levels: int — count of nested sub-problem dirs
    n_sampling_attempts: int — sum of items across failure pickles
    has_patch_attempts: bool — any hint_repair/ dirs

Both formats are tolerated when subsets are incomplete (in-progress rollouts).
"""

from __future__ import annotations
import json
import os
import glob
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any


ROLLOUT_ROOT = "outputs/rollouts/gemini3flash"
FLAT_ROOT = "outputs/flat_sampling"


# ---------------------------------------------------------------------------
# Rollout (GammaZero) loader
# ---------------------------------------------------------------------------

def _summarize_rollout(d: dict) -> dict:
    nodes = d.get("nodes", [])
    or_nodes = [n for n in nodes if n.get("type") == "OR"]
    and_nodes = [n for n in nodes if n.get("type") == "AND"]
    skel_and = [n for n in and_nodes if n.get("action_type") == "skeleton"]
    tact_and = [n for n in and_nodes if n.get("action_type") == "tactic"]

    meta = d.get("search_metadata", {})
    fs = meta.get("final_status", {}) or {}
    states = fs.get("states", {}) or {}
    actions = fs.get("actions", {}) or {}
    dd = meta.get("depth_distribution", {}) or {}
    states_solved_by_depth = dd.get("states_solved_by_depth", {}) or {}
    max_depth_reached = int(dd.get("max_depth_reached", 0) or 0)

    root_solved = int(states.get("SOLVED", 0)) > 0  # OR root reaches SOLVED
    # Determine if the root was solved purely at depth 0 (no skeleton expansion needed)
    solved_at_depth_0 = int(states_solved_by_depth.get("0", 0) or 0) > 0
    solved_via_skel = root_solved and (max_depth_reached > 0)

    budget = meta.get("budget", {}) or {}
    skel_pipe = meta.get("skeleton_pipeline", {}) or {}
    skel_commit = meta.get("skeleton_commitment", {}) or {}
    beam = meta.get("beam", {}) or {}

    return {
        "metadata": meta,
        "nodes": nodes,
        "edges": d.get("edges", []),
        "n_nodes": len(nodes),
        "n_or": len(or_nodes),
        "n_and": len(and_nodes),
        "n_skeleton_and": len(skel_and),
        "n_tactic_and": len(tact_and),
        "solved": root_solved,
        "solved_root_only": root_solved and not solved_via_skel and solved_at_depth_0,
        "solved_via_skeleton": solved_via_skel,
        "max_depth_reached": max_depth_reached,
        "used_total": int(budget.get("used_total", 0) or 0),
        "used_tactic": int(budget.get("used_tactic", 0) or 0),
        "used_skeleton_raw": int(budget.get("used_skeleton_raw", 0) or 0),
        "lean_verify_calls": int(budget.get("lean_verify_calls", 0) or 0),
        "patch_verify_calls": int(budget.get("patch_verify_calls", 0) or 0),
        "tactic_solved": int(actions.get("tactic_SOLVED", 0) or 0),
        "tactic_failed": int(actions.get("tactic_FAILED", 0) or 0),
        "skeleton_solved": int(actions.get("skeleton_SOLVED", 0) or 0),
        "skeleton_failed": int(actions.get("skeleton_FAILED", 0) or 0),
        "skeleton_reserved": int(actions.get("skeleton_RESERVED", 0) or 0),
        "skeleton_pipeline": skel_pipe,
        "skeleton_commitment": skel_commit,
        "beam": beam,
        "states_solved_by_depth": states_solved_by_depth,
    }


def load_rollout_file(path: str, keep_raw: bool = True) -> dict:
    with open(path) as f:
        d = json.load(f)
    summary = _summarize_rollout(d)
    rec = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "split": _detect_split(path),
        "system": "gammazero",
        "path": path,
        "has_data": True,
        **summary,
    }
    if not keep_raw:
        rec.pop("nodes", None)
        rec.pop("edges", None)
        rec.pop("metadata", None)
    return rec


def _detect_split(path: str) -> str:
    p = path.lower()
    if "valid" in p:
        return "valid"
    if "test" in p:
        return "test"
    return "unknown"


def load_rollouts(root: str = ROLLOUT_ROOT, keep_raw: bool = True) -> list[dict]:
    """Load all GammaZero rollout JSONs under `root`."""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    records = []
    for p in paths:
        try:
            records.append(load_rollout_file(p, keep_raw=keep_raw))
        except Exception as e:
            records.append({
                "name": os.path.splitext(os.path.basename(p))[0],
                "split": _detect_split(p),
                "system": "gammazero",
                "path": p,
                "has_data": False,
                "error": str(e),
                "solved": False,
            })
    return records


# ---------------------------------------------------------------------------
# Flat sampling (Kimina) loader
# ---------------------------------------------------------------------------

def _kimina_problem_dirs(root: str) -> list[str]:
    """Top-level problem dirs under flat_sampling/<bench>/."""
    # Bench dirs (e.g., miniF2F-Test)
    bench_dirs = [
        d for d in glob.glob(os.path.join(root, "*"))
        if os.path.isdir(d)
    ]
    out = []
    for b in bench_dirs:
        for p in sorted(glob.glob(os.path.join(b, "*"))):
            if os.path.isdir(p):
                out.append(p)
    return out


def _count_recursion_levels(problem_dir: str) -> int:
    """Count nested sub-problem directories — proxy for recursive APOLLO depth."""
    cur = problem_dir
    depth = 0
    while True:
        children = [
            d for d in glob.glob(os.path.join(cur, "*"))
            if os.path.isdir(d) and not d.endswith("hint_repair") and not d.endswith("run0")
        ]
        if not children:
            return depth
        cur = children[0]
        depth += 1
        if depth > 16:  # safety
            return depth


def _count_failure_samples(problem_dir: str) -> int:
    total = 0
    for pkl in glob.glob(os.path.join(problem_dir, "**", "failure-Sampling-*.pkl"), recursive=True):
        try:
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            if hasattr(d, "__len__"):
                total += len(d)
        except Exception:
            pass
    return total


def load_kimina_problem(problem_dir: str) -> dict:
    name = os.path.basename(problem_dir)
    bench = os.path.basename(os.path.dirname(problem_dir))
    asm = os.path.join(problem_dir, "assembled_main_theorem.lean")
    solved = False
    has_assembly = os.path.exists(asm)
    if has_assembly:
        try:
            body = open(asm).read()
            solved = ("sorry" not in body) and (":=" in body)
        except Exception:
            pass
    n_rec = _count_recursion_levels(problem_dir)
    n_samples = _count_failure_samples(problem_dir)
    has_repair = bool(glob.glob(os.path.join(problem_dir, "**", "hint_repair"), recursive=True))
    finished = bool(glob.glob(os.path.join(problem_dir, "**", "finished_running.txt"), recursive=True))
    return {
        "name": name,
        "split": "valid" if "valid" in bench.lower() else ("test" if "test" in bench.lower() else "unknown"),
        "system": "kimina",
        "path": problem_dir,
        "has_data": has_assembly,
        "solved": solved,
        "n_recursion_levels": n_rec,
        "n_sampling_attempts": n_samples,
        "has_patch_attempts": has_repair,
        "finished_running": finished,
    }


def load_kimina(root: str = FLAT_ROOT) -> list[dict]:
    return [load_kimina_problem(p) for p in _kimina_problem_dirs(root)]


# ---------------------------------------------------------------------------
# Convenience: align problems across systems
# ---------------------------------------------------------------------------

def align_by_name(gz: list[dict], km: list[dict]) -> list[dict]:
    """Return per-problem joined records (by lowercase name)."""
    km_idx = {r["name"].lower(): r for r in km}
    out = []
    for g in gz:
        k = km_idx.get(g["name"].lower())
        out.append({
            "name": g["name"],
            "split": g["split"],
            "gz": g,
            "kimina": k,
            "gz_solved": bool(g.get("solved")),
            "kimina_solved": bool(k.get("solved")) if k else None,
            "both": bool(g.get("solved")) and bool(k.get("solved") if k else False),
            "only_gz": bool(g.get("solved")) and not bool(k.get("solved") if k else False),
            "only_kimina": (not bool(g.get("solved"))) and bool(k.get("solved") if k else False),
            "neither": (not bool(g.get("solved"))) and (not bool(k.get("solved") if k else False)),
        })
    return out
