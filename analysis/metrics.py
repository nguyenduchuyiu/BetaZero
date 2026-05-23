"""Metrics computed from loaded rollout / kimina records.

All functions take iterables of records (from loaders.py) and return either a
single dict of aggregate stats or a list of per-record dicts. Designed to be
order-independent and tolerant of in-progress / partial data.
"""

from __future__ import annotations
import math
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Iterable


# ---------------------------------------------------------------------------
# Simple aggregates
# ---------------------------------------------------------------------------

def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else float("nan")


def _safe_median(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else float("nan")


def _safe_std(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else 0.0


def solve_rate(records: Iterable[dict]) -> dict:
    recs = [r for r in records if r.get("has_data")]
    n = len(recs)
    solved = sum(1 for r in recs if r.get("solved"))
    return {
        "n": n,
        "solved": solved,
        "rate": (solved / n) if n else 0.0,
    }


def solve_rate_by_split(records: Iterable[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("split", "unknown")].append(r)
    return {k: solve_rate(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# GammaZero-specific metrics
# ---------------------------------------------------------------------------

def hierarchical_breakdown(gz_records: Iterable[dict]) -> dict:
    """Of solved problems, how many used the hierarchy?"""
    solved = [r for r in gz_records if r.get("has_data") and r.get("solved")]
    total_with_data = sum(1 for r in gz_records if r.get("has_data"))
    root_only = sum(1 for r in solved if r.get("solved_root_only"))
    via_skel = sum(1 for r in solved if r.get("solved_via_skeleton"))
    other = len(solved) - root_only - via_skel  # solved at depth 0 but also expanded
    return {
        "total_with_data": total_with_data,
        "solved": len(solved),
        "solved_root_only": root_only,
        "solved_via_skeleton": via_skel,
        "solved_other": other,
        "root_only_rate": (root_only / total_with_data) if total_with_data else 0.0,
        "skeleton_marginal_gain": (via_skel / total_with_data) if total_with_data else 0.0,
    }


def depth_of_solution(gz_records: Iterable[dict]) -> dict:
    """Histogram of max_depth_reached for solved vs all problems."""
    solved_depths = []
    all_depths = []
    for r in gz_records:
        if not r.get("has_data"):
            continue
        d = int(r.get("max_depth_reached", 0) or 0)
        all_depths.append(d)
        if r.get("solved"):
            solved_depths.append(d)
    hist_all = defaultdict(int)
    hist_solved = defaultdict(int)
    for d in all_depths:
        hist_all[d] += 1
    for d in solved_depths:
        hist_solved[d] += 1
    return {
        "hist_all": dict(sorted(hist_all.items())),
        "hist_solved": dict(sorted(hist_solved.items())),
        "mean_depth_all": _safe_mean(all_depths),
        "mean_depth_solved": _safe_mean(solved_depths),
        "max_depth_observed": max(all_depths) if all_depths else 0,
    }


def search_cost(gz_records: Iterable[dict]) -> dict:
    rs = [r for r in gz_records if r.get("has_data")]
    fields = [
        "n_nodes", "n_or", "n_and",
        "n_tactic_and", "n_skeleton_and",
        "used_total", "used_tactic", "used_skeleton_raw",
        "lean_verify_calls", "patch_verify_calls",
    ]
    out = {}
    for f in fields:
        vals = [r.get(f, 0) or 0 for r in rs]
        out[f] = {
            "mean": _safe_mean(vals),
            "median": _safe_median(vals),
            "std": _safe_std(vals),
            "max": max(vals) if vals else 0,
            "sum": sum(vals),
        }
    # Also broken down solved vs failed
    solved = [r for r in rs if r.get("solved")]
    failed = [r for r in rs if not r.get("solved")]
    out["solved_vs_failed"] = {
        "n_nodes_solved_mean": _safe_mean([r["n_nodes"] for r in solved]),
        "n_nodes_failed_mean": _safe_mean([r["n_nodes"] for r in failed]),
        "lean_calls_solved_mean": _safe_mean([r["lean_verify_calls"] for r in solved]),
        "lean_calls_failed_mean": _safe_mean([r["lean_verify_calls"] for r in failed]),
    }
    return out


def skeleton_funnel(gz_records: Iterable[dict]) -> dict:
    """Aggregate skeleton-pipeline counters across problems."""
    keys = [
        "requested", "raw_verify_success", "raw_verify_failed",
        "patch_attempted", "patch_scored", "patch_failed", "feedback_generated",
        "inserted_raw", "selected_by_beam", "rejected_by_beam",
        "valid_zero_children", "skeleton_duplicate_actions",
        "children_new", "children_duplicate",
    ]
    sums = {k: 0 for k in keys}
    for r in gz_records:
        sp = r.get("skeleton_pipeline", {}) or {}
        for k in keys:
            sums[k] += int(sp.get(k, 0) or 0)
    commit_keys = ["committed", "reserved", "fallback_activated",
                   "committed_solved", "committed_failed", "committed_stale",
                   "blocked_new_skeleton_due_to_active_commit"]
    commit_sums = {k: 0 for k in commit_keys}
    for r in gz_records:
        sc = r.get("skeleton_commitment", {}) or {}
        for k in commit_keys:
            commit_sums[k] += int(sc.get(k, 0) or 0)
    funnel_rates = {}
    if sums["requested"]:
        funnel_rates["raw_verify_success_rate"] = sums["raw_verify_success"] / sums["requested"]
        funnel_rates["committed_rate"] = commit_sums["committed"] / sums["requested"]
    if sums["patch_attempted"]:
        funnel_rates["patch_score_rate"] = sums["patch_scored"] / sums["patch_attempted"]
    if commit_sums["committed"]:
        funnel_rates["committed_solved_rate"] = commit_sums["committed_solved"] / commit_sums["committed"]
    return {"pipeline": sums, "commitment": commit_sums, "rates": funnel_rates}


# ---------------------------------------------------------------------------
# AND-node level: reward distributions & separability
# ---------------------------------------------------------------------------

def and_node_metrics(gz_records: Iterable[dict]) -> list[dict]:
    """Flatten every AND node into a row: (problem, action_type, status, r_env, r_dep, Q, depth)."""
    rows = []
    for r in gz_records:
        if not r.get("nodes"):
            continue
        name = r["name"]
        # Build OR depth lookup
        or_depth = {}
        for n in r["nodes"]:
            if n.get("type") == "OR":
                or_depth[n.get("id")] = int(n.get("depth", 0) or 0)
        # Build child→parent OR via edges (AND has source as its OR parent)
        and_parent_or = {}
        for e in r.get("edges", []):
            if e.get("relation") == "expanded_to":
                # source is OR, target is AND
                and_parent_or[e.get("target")] = e.get("source")
        for n in r["nodes"]:
            if n.get("type") != "AND":
                continue
            m = n.get("metrics") or {}
            parent_or = and_parent_or.get(n.get("id"))
            depth = or_depth.get(parent_or, None)
            rows.append({
                "problem": name,
                "split": r.get("split"),
                "and_id": n.get("id"),
                "action_type": n.get("action_type"),
                "status": n.get("status"),
                "r_env": _f(m.get("r_env")),
                "r_dep": _f(m.get("r_dep")),
                "Q": _f(m.get("Q_value")),
                "depth": depth,
                "solved": n.get("status") == "SOLVED",
                "failed": n.get("status") == "FAILED",
            })
    return rows


def _f(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def reward_distribution(and_rows: list[dict]) -> dict:
    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": mean(xs),
            "median": median(xs),
            "std": pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    out = {}
    for component in ("r_env", "r_dep", "Q"):
        for action in (None, "tactic", "skeleton"):
            for st in (None, "SOLVED", "FAILED", "RESERVED"):
                key = f"{component}|{action or 'all'}|{st or 'all'}"
                vals = [
                    r[component] for r in and_rows
                    if (action is None or r["action_type"] == action)
                    and (st is None or r["status"] == st)
                ]
                out[key] = stats(vals)
    return out


def reward_separability(and_rows: list[dict]) -> dict:
    """Mann-Whitney-like AUROC of reward components separating SOLVED vs FAILED AND nodes."""
    out = {}
    for component in ("r_env", "r_dep", "Q"):
        for action in (None, "tactic", "skeleton"):
            pos = [r[component] for r in and_rows
                   if r["status"] == "SOLVED"
                   and r[component] is not None
                   and (action is None or r["action_type"] == action)]
            neg = [r[component] for r in and_rows
                   if r["status"] == "FAILED"
                   and r[component] is not None
                   and (action is None or r["action_type"] == action)]
            key = f"{component}|{action or 'all'}"
            out[key] = {
                "n_pos": len(pos),
                "n_neg": len(neg),
                "auroc": _auroc(pos, neg),
                "delta_mean": (mean(pos) - mean(neg)) if pos and neg else float("nan"),
            }
    return out


def _auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney U → AUROC. Returns NaN if either group empty."""
    if not pos or not neg:
        return float("nan")
    # rank all values, average rank in ties
    combined = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(ranks[k] for k in range(n) if combined[k][1] == 1)
    n_p, n_n = len(pos), len(neg)
    U = sum_pos_ranks - n_p * (n_p + 1) / 2
    return U / (n_p * n_n)


# ---------------------------------------------------------------------------
# Outcome correlation: per-problem root-Q vs eventual solve
# ---------------------------------------------------------------------------

def root_q_vs_outcome(gz_records: Iterable[dict]) -> dict:
    """For each problem: best Q among children of the root OR. Correlate with solve."""
    pairs = []
    for r in gz_records:
        if not r.get("nodes"):
            continue
        # Root OR id is the one with depth==0 and earliest id (often "root_goal")
        ors = [n for n in r["nodes"] if n.get("type") == "OR" and int(n.get("depth", 0) or 0) == 0]
        if not ors:
            continue
        root_ids = {n.get("id") for n in ors}
        # Children AND nodes via edges
        and_children = []
        for e in r.get("edges", []):
            if e.get("relation") == "expanded_to" and e.get("source") in root_ids:
                and_children.append(e.get("target"))
        and_id_set = set(and_children)
        qs = []
        for n in r["nodes"]:
            if n.get("type") == "AND" and n.get("id") in and_id_set:
                q = _f((n.get("metrics") or {}).get("Q_value"))
                if q is not None:
                    qs.append(q)
        if not qs:
            continue
        pairs.append((max(qs), bool(r.get("solved"))))
    if not pairs:
        return {"n": 0, "auroc": float("nan")}
    pos = [q for q, s in pairs if s]
    neg = [q for q, s in pairs if not s]
    return {
        "n": len(pairs),
        "auroc": _auroc(pos, neg),
        "mean_q_solved": _safe_mean(pos),
        "mean_q_failed": _safe_mean(neg),
    }


# ---------------------------------------------------------------------------
# Trajectory dataset stats (RL-data framing)
# ---------------------------------------------------------------------------

def trajectory_stats(gz_records: Iterable[dict]) -> dict:
    """Yield of RL transitions: count SOLVED AND nodes (positive transitions) and
    FAILED AND nodes (negative transitions) per problem and globally."""
    pos_trans, neg_trans, reserved = 0, 0, 0
    per_problem = []
    for r in gz_records:
        if not r.get("nodes"):
            continue
        p, n_, rsv = 0, 0, 0
        for n in r["nodes"]:
            if n.get("type") != "AND":
                continue
            s = n.get("status")
            if s == "SOLVED":
                p += 1
            elif s == "FAILED":
                n_ += 1
            elif s == "RESERVED":
                rsv += 1
        pos_trans += p
        neg_trans += n_
        reserved += rsv
        per_problem.append({"name": r["name"], "pos": p, "neg": n_, "reserved": rsv,
                            "solved": r.get("solved")})
    n_problems = len(per_problem)
    return {
        "n_problems": n_problems,
        "total_pos_transitions": pos_trans,
        "total_neg_transitions": neg_trans,
        "total_reserved": reserved,
        "pos_per_problem_mean": pos_trans / n_problems if n_problems else 0.0,
        "neg_per_problem_mean": neg_trans / n_problems if n_problems else 0.0,
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Cross-system comparison
# ---------------------------------------------------------------------------

def system_comparison(joined: Iterable[dict]) -> dict:
    """Counts of both/only-gz/only-kimina/neither over the joined view."""
    both = only_gz = only_km = neither = 0
    no_km = 0
    for r in joined:
        if r["kimina"] is None or not r["kimina"].get("has_data"):
            no_km += 1
            continue
        if r["both"]:
            both += 1
        elif r["only_gz"]:
            only_gz += 1
        elif r["only_kimina"]:
            only_km += 1
        else:
            neither += 1
    n = both + only_gz + only_km + neither
    return {
        "n_aligned": n,
        "n_gz_only_in_set": no_km,
        "both": both,
        "only_gammazero": only_gz,
        "only_kimina": only_km,
        "neither": neither,
        "gammazero_unique_solves": only_gz,
        "kimina_unique_solves": only_km,
    }


def kimina_aggregates(km_records: Iterable[dict]) -> dict:
    rs = [r for r in km_records if r.get("has_data")]
    n = len(rs)
    solved = sum(1 for r in rs if r.get("solved"))
    rec_levels = [int(r.get("n_recursion_levels", 0) or 0) for r in rs]
    samples = [int(r.get("n_sampling_attempts", 0) or 0) for r in rs]
    return {
        "n_with_data": n,
        "solved": solved,
        "rate": (solved / n) if n else 0.0,
        "recursion_levels": {
            "mean": _safe_mean(rec_levels),
            "median": _safe_median(rec_levels),
            "max": max(rec_levels) if rec_levels else 0,
        },
        "sampling_attempts": {
            "mean": _safe_mean(samples),
            "median": _safe_median(samples),
            "sum": sum(samples),
        },
    }
