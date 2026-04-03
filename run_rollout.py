import argparse
import datetime as _dt
import gc
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from betazero.core.nodes import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.search.and_or_graph import ANDORGraph
from betazero.search.rollout import LevelwiseRollout
from betazero.search.sorrifier import Sorrifier
from betazero.search.reward import RewardCalculator
from betazero.utils.dataloader import parse_lean_file
from betazero.utils.config import Config
from betazero.policy.vllm_process import VLLMProcess


class MockPolicy:
    """Policy mock để sinh hành động 'sorry' cho mục đích test rollout/graph."""

    def __init__(self, tactic_code: str = "sorry", skeleton_code: str = "sorry"):
        self.tactic_code = tactic_code
        self.skeleton_code = skeleton_code

    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]:
        if not states:
            return []
        code = self.tactic_code if action_type == "tactic" else self.skeleton_code
        return [[code] * n for _ in states]


def _find_graph(root_state: ProofState) -> Optional[ANDORGraph]:
    for obj in gc.get_objects():
        if isinstance(obj, ANDORGraph) and len(obj.get_actions(root_state)) > 0:
            return obj
    return None


def _tree_to_lines(graph: ANDORGraph, root: ProofState) -> list[str]:
    lines: list[str] = []
    visited: set[ProofState] = set()

    def node_line(prefix: str, s: ProofState) -> str:
        goal_head = (s.goal.splitlines()[0] if s.goal else "").strip()
        return f"{prefix}STATE goal={goal_head!r}"

    def action_lines(prefix: str, a: Action, i: int) -> list[str]:
        status = "SOLVED" if graph.is_solved(a) else "OPEN"
        r_env = graph.get_r_env(a)
        out = [
            f"{prefix}ACTION#{i} type={a.action_type} status={status} r_env={r_env:.3f}"
        ]
        body = a.content.strip()
        if body:
            for line in body.splitlines():
                out.append(f"{prefix}  {line}")
        else:
            out.append(f"{prefix}  (empty)")
        return out

    def rec(state: ProofState, depth: int):
        indent = "  " * depth
        if state in visited:
            lines.append(f"{indent}(visited) {node_line('', state)}")
            return
        visited.add(state)

        lines.append(node_line(f"{indent}", state))
        actions = graph.get_actions(state)
        if not actions:
            lines.append(f"{indent}  (no actions)")
            return

        for i, act in enumerate(actions, 1):
            lines.extend(action_lines(f"{indent}  ", act, i))
            for child in act.children:
                rec(child, depth + 2)

    rec(root, 0)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--lean-file", type=str, default="problems/aime_1984_p5.lean")
    ap.add_argument("--out-md", type=str, default="outputs/rollout_aime_1984_p5.md")
    ap.add_argument("--policy", type=str, choices=["mock", "vllm"], default="mock")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--lean-timeout", type=int, default=None)
    args = ap.parse_args()

    cfg: Config = Config()
    if args.config:
        cfg = Config.from_yaml(args.config)

    lean_path = os.path.join(os.path.dirname(__file__), args.lean_file)
    if not os.path.exists(lean_path):
        raise FileNotFoundError(lean_path)

    root_state = parse_lean_file(lean_path)
    if root_state is None:
        raise RuntimeError(f"Could not parse theorem from: {lean_path}")

    K = args.K if args.K is not None else cfg.K
    max_depth = args.max_depth if args.max_depth is not None else cfg.max_depth
    max_nodes = args.max_nodes if args.max_nodes is not None else cfg.max_nodes
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else cfg.lean_timeout

    verifier = Lean4ServerScheduler(
        max_concurrent_requests=cfg.lean_workers, timeout=lean_timeout, name="rollout_external"
    )
    lean = LeanEnv(verifier)
    sorrifier = Sorrifier(verifier)
    reward = RewardCalculator()

    vllm = None
    try:
        if args.policy == "mock":
            policy = MockPolicy()
        else:
            # Use cfg loaded from --config for model/vLLM settings.
            vllm = VLLMProcess(cfg)
            vllm.start(args.adapter)
            policy = vllm

        rollout = LevelwiseRollout(
            policy=policy,
            lean=lean,
            sorrifier=sorrifier,
            reward=reward,
            K=K,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        _ = rollout.rollout(root_state)
        graph = _find_graph(root_state)
        if graph is None:
            raise RuntimeError("Could not locate ANDORGraph instance after rollout.")

        tree_lines = _tree_to_lines(graph, root_state)
        now = _dt.datetime.now().isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Rollout report\n")
            f.write(f"- time: {now}\n")
            f.write(f"- lean file: `{args.lean_file}`\n") 
            if args.config:
                f.write(f"- config: `{args.config}`\n")
            f.write(f"- policy: `{args.policy}`\n")
            f.write(f"- params: K={K}, max_depth={max_depth}, max_nodes={max_nodes}, lean_timeout={lean_timeout}\n")
            f.write("\n")
            f.write("## Root theorem\n")
            f.write(f"### Context\n```\n{root_state.context}\n```\n")
            f.write(f"### Goal\n```\n{root_state.goal}\n```\n")
            if root_state.header:
                f.write(f"### Header\n```\n{root_state.header}\n```\n")
            f.write("\n")
            f.write("## AND/OR tree\n")
            f.write("```text\n")
            f.write("\n".join(tree_lines))
            f.write("\n```\n")

    finally:
        if vllm is not None:
            vllm.kill()
        verifier.close()


if __name__ == "__main__":
    main()

#python3 run_rollout.py   --config configs/qwen2.5_0.5b.yaml   --lean-file problems/aime_1984_p5.lean   --out-md outputs/rollout_aime_1984_p5.md   --policy vllm
