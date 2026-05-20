from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gammazero.core import Action, ProofState
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import DependencyRewardAssigner, RewardCalculator
from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.utils.graph_logger import GraphLogger


DEFAULT_JSON = Path(
    "outputs/rollouts/gemini3flash/miniF2F-valid-aime/aime_1987_p5.json"
)


def has_real_placeholder(code: str, name: str) -> bool:
    clean = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")
    return bool(re.search(rf"\b{name}\b", clean))


def build_graph_from_rollout(data: dict) -> tuple[ANDORGraph, ProofState]:
    node_by_id = {node["id"]: node for node in data.get("nodes", [])}

    state_to_actions: dict[str, list[str]] = {}
    action_to_children: dict[str, list[str]] = {}
    action_to_parent: dict[str, str] = {}
    for edge in data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        if edge.get("relation") == "expanded_to":
            state_to_actions.setdefault(src, []).append(tgt)
            action_to_parent[tgt] = src
        elif edge.get("relation") == "subgoal":
            action_to_children.setdefault(src, []).append(tgt)

    state_objs: dict[str, ProofState] = {}
    for node_id, node in node_by_id.items():
        if node.get("type") != "OR":
            continue
        content = node.get("content") or {}
        state_objs[node_id] = ProofState(
            context=content.get("context", ""),
            goal=content.get("goal", ""),
            header="",
            scaffold_code=content.get("scaffold_code") or "",
            target_index=content.get("target_index") or 0,
            target_kind=content.get("target_kind") or "",
            parent_action_id=content.get("parent_action_id") or "",
        )

    root_id = data.get("root_id")
    if root_id not in state_objs:
        raise ValueError(f"root_id is missing or not an OR node: {root_id!r}")

    root = state_objs[root_id]
    graph = ANDORGraph(root)
    for state_id, state in state_objs.items():
        if state_id == root_id:
            continue
        graph.add_state(state, depth=(node_by_id[state_id].get("depth") or 0))

    for action_id, parent_id in action_to_parent.items():
        node = node_by_id[action_id]
        if node.get("type") != "AND":
            continue
        parent = state_objs[parent_id]
        children = tuple(state_objs[cid] for cid in action_to_children.get(action_id, []))
        action = Action(
            action_type=node.get("action_type", "tactic"),
            content=node.get("content") or "",
            extracted_code=node.get("extracted_lean_code") or "",
            children=children,
            prompt=node.get("prompt") or "",
            verify_code=node.get("verify_code") or "",
            stitched_code=node.get("stitched_code") or "",
            patched_code=node.get("patched_code") or "",
            lean_feedback=node.get("lean_feedback") or "",
            target_child_index=node.get("target_child_index"),
            id=node.get("internal_id") or action_id,
        )
        metrics = node.get("metrics") or {}
        tactic_status = None
        if action.action_type == "tactic":
            tactic_status = "SOLVED" if node.get("status") == "SOLVED" else "FAILED"
        graph.expand(
            parent,
            action,
            r_env=metrics.get("r_env", 0.0),
            r_dep=metrics.get("r_dep", 0.0),
            tactic_status=tactic_status,
        )
        if node.get("status") == "FAILED":
            graph.mark_failed(action)
        elif node.get("status") == "RESERVED":
            graph.mark_reserved(action)

    return graph, root


def restitch_rollout_json(
    input_path: Path,
    output_path: Path,
    *,
    workers: int,
    timeout: int,
) -> None:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    graph, root = build_graph_from_rollout(data)

    scheduler = Lean4ServerScheduler(
        max_concurrent_requests=workers,
        timeout=timeout,
        name="restitch",
    )
    try:
        lean = LeanEnv(scheduler)
        reward = RewardCalculator()
        assigner = DependencyRewardAssigner(lean, reward)
        assigner.stitch_and_score_skeletons(graph)
        q_values = reward.compute_returns(graph)
        out = GraphLogger().export_to_dict(graph, root, q_values)
    finally:
        scheduler.close()

    proof = next(
        node
        for node in out["nodes"]
        if node["id"] == out["root_id"]
    )["content"].get("proof_body", "")
    sorry_count = len(re.findall(r"\bsorry\b", re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", proof or "")))
    admit_count = len(re.findall(r"\badmit\b", re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", proof or "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote: {output_path}")
    print(f"root proof chars: {len(proof)}")
    print(f"root proof sorry count: {sorry_count}")
    print(f"root proof admit count: {admit_count}")


def fast_restitch_rollout_json(input_path: Path, output_path: Path) -> None:
    """Restitch from the solved proof bodies already stored in the rollout JSON.

    This does not call Lean or recompute dependency rewards. It is meant for
    debugging stale parent proof bodies in the graph viewer.
    """
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    node_by_id = {node["id"]: node for node in data.get("nodes", [])}
    state_to_actions: dict[str, list[str]] = {}
    action_to_children: dict[str, list[str]] = {}
    action_to_parent: dict[str, str] = {}
    for edge in data.get("edges", []):
        if edge.get("relation") == "expanded_to":
            state_to_actions.setdefault(edge["source"], []).append(edge["target"])
            action_to_parent[edge["target"]] = edge["source"]
        elif edge.get("relation") == "subgoal":
            action_to_children.setdefault(edge["source"], []).append(edge["target"])

    visiting_states: set[str] = set()
    visiting_actions: set[str] = set()
    state_cache: dict[str, tuple[bool, str]] = {}
    action_cache: dict[str, tuple[bool, str]] = {}
    updated = 0

    def set_field(node: dict, key: str, value) -> None:
        nonlocal updated
        if node.get(key) != value:
            node[key] = value
            updated += 1

    def set_content_field(node: dict, key: str, value) -> None:
        nonlocal updated
        content = node.setdefault("content", {})
        if content.get(key) != value:
            content[key] = value
            updated += 1

    def eval_action(action_id: str) -> tuple[bool, str]:
        if action_id in action_cache:
            return action_cache[action_id]
        if action_id in visiting_actions:
            return False, "  sorry"
        visiting_actions.add(action_id)

        node = node_by_id[action_id]
        if node.get("action_type") == "tactic":
            solved = node.get("status") == "SOLVED"
            proof = node.get("extracted_lean_code") or "  sorry"
            result = (solved, proof)
            action_cache[action_id] = result
            visiting_actions.remove(action_id)
            return result

        child_ids = action_to_children.get(action_id, [])
        child_results = [eval_state(child_id) for child_id in child_ids]
        child_proofs = [proof if solved else None for solved, proof in child_results]
        extracted = node.get("extracted_lean_code") or ""
        stitched = ProofStitcher.stitch(extracted, child_proofs) if extracted else "  sorry"
        solved = bool(child_ids) and all(solved for solved, _ in child_results)

        set_field(node, "stitched_code", stitched)
        if solved:
            set_field(node, "status", "SOLVED")
        result = (solved, stitched if solved else "  sorry")
        action_cache[action_id] = result
        visiting_actions.remove(action_id)
        return result

    def eval_state(state_id: str) -> tuple[bool, str]:
        if state_id in state_cache:
            return state_cache[state_id]
        if state_id in visiting_states:
            return False, "  sorry"
        visiting_states.add(state_id)

        candidates = []
        for action_id in state_to_actions.get(state_id, []):
            solved, proof = eval_action(action_id)
            if solved:
                candidates.append((action_id, proof))

        if candidates:
            clean = [(aid, proof) for aid, proof in candidates if not has_real_placeholder(proof, "sorry")]
            _, proof = (clean or candidates)[0]
            set_field(node_by_id[state_id], "status", "SOLVED")
            set_content_field(node_by_id[state_id], "proof_body", proof)
            result = True, proof
        else:
            result = False, "  sorry"

        state_cache[state_id] = result
        visiting_states.remove(state_id)
        return result

    root_id = data.get("root_id")
    if not root_id:
        raise ValueError("missing root_id")
    root_solved, root_proof = eval_state(root_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    clean_root = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", root_proof or "")
    print(f"wrote: {output_path}")
    print(f"updated fields: {updated}")
    print(f"root solved by logged graph: {root_solved}")
    print(f"root proof chars: {len(root_proof or '')}")
    print(f"root proof sorry count: {len(re.findall(r'\\bsorry\\b', clean_root))}")
    print(f"root proof admit count: {len(re.findall(r'\\badmit\\b', clean_root))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a rollout graph from JSON and restitch proofs.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    output_path = args.out
    if args.in_place:
        output_path = args.json
    elif output_path is None:
        output_path = args.json.with_suffix(".restitched.json")

    if args.fast:
        fast_restitch_rollout_json(args.json, output_path)
    else:
        restitch_rollout_json(
            args.json,
            output_path,
            workers=max(1, args.workers),
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
