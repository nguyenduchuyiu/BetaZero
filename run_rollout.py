import argparse
import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from gammazero.core.nodes import ProofState, Action
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.graph.and_or_graph import ANDORGraph
from gammazero.search.rollout.best_first_rollout import BestFirstRollout
from gammazero.search.rollout.heuristic import SimpleHeuristicScorer
from gammazero.search.sorrifier import Sorrifier
from gammazero.search.reward import RewardCalculator
from gammazero.utils.config import Config
from gammazero.policy.vllm_server import VLLMServer
from gammazero.policy.deepseek_server import DeepSeekAPIServer
from gammazero.policy.gemini_server import GeminiAPIServer
from gammazero.policy.prompt import SearchPromptBuilder
from gammazero.utils.lean_parse import parse_proof_state
from gammazero.utils.lean_cmd import DEFAULT_OPEN
from gammazero.utils.graph_logger import GraphLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--lean-file", type=str, default=None)
    ap.add_argument("--lean-dir", type=str, default=None)
    ap.add_argument("--out-json", type=str, default="outputs/rollouts/")
    ap.add_argument("--adapter", type=str, default="adapter-deepseek-qwen-7b")
    ap.add_argument("--policy", type=str, choices=["vllm", "deepseek", "gemini"], default="vllm")
    args = ap.parse_args()

    if not args.lean_file and not args.lean_dir:
        args.lean_file = "problems/aime_1984_p5.lean"

    cfg: Config = Config()
    if args.config:
        cfg = Config.from_yaml(args.config)

    max_depth = cfg.max_depth
    max_nodes = cfg.max_nodes
    lean_timeout = cfg.lean_timeout

    lean_files = []
    if args.lean_dir:
        if not os.path.exists(args.lean_dir):
            raise FileNotFoundError(args.lean_dir)
        if not os.path.isdir(args.lean_dir):
            raise NotADirectoryError(
                f"--lean-dir expects a directory, got file: {args.lean_dir}. "
                "Use --lean-file for a single Lean file."
            )
        for root, _, files in os.walk(args.lean_dir):
            for f in files:
                if f.endswith(".lean"):
                    lean_files.append(os.path.abspath(os.path.join(root, f)))
        lean_files.sort()
        if not lean_files:
            raise FileNotFoundError(f"No .lean files found under --lean-dir: {args.lean_dir}")
    else:
        lean_path = os.path.abspath(args.lean_file)
        if not os.path.exists(lean_path):
            raise FileNotFoundError(lean_path)
        lean_files = [lean_path]

    verifier = Lean4ServerScheduler(
        max_concurrent_requests=cfg.lean_workers, timeout=lean_timeout, name="rollout_external"
    )
    lean = LeanEnv(verifier)
    sorrifier = Sorrifier(verifier, max_cycles=cfg.sorrifier_max_cycles)
    reward = RewardCalculator()

    policy_server = None
    try:
        if args.policy == "vllm":
            policy_server = VLLMServer(cfg)
        elif args.policy == "deepseek":
            policy_server = DeepSeekAPIServer(cfg)
        elif args.policy == "gemini":
            policy_server = GeminiAPIServer(cfg)
        else:
            raise ValueError(f"Unknown policy: {args.policy}")
            
        policy_server.start(args.adapter)
        policy = policy_server

        rollout_engine = BestFirstRollout(
            policy=policy,
            lean=lean,
            sorrifier=sorrifier,
            reward=reward,
            max_depth=max_depth,
            max_nodes=max_nodes,
            search_batch_size=cfg.search_batch_size,
            initial_tactic_k=cfg.initial_tactic_k,
            retry_tactic_k=cfg.retry_tactic_k,
            max_tactic_per_state=cfg.max_tactic_per_state,
            initial_skeleton_k=cfg.initial_skeleton_k,
            retry_skeleton_k=cfg.retry_skeleton_k,
            max_skeleton_per_state=cfg.max_skeleton_per_state,
            state_beam_width=cfg.state_beam_width,
            state_beam_per_depth=cfg.state_beam_per_depth,
            skeleton_beam_per_state=cfg.skeleton_beam_per_state,
            skeleton_commitment=cfg.skeleton_commitment,
            max_reserved_skeletons_per_state=cfg.max_reserved_skeletons_per_state,
            commit_stale_rounds_before_fallback=cfg.commit_stale_rounds_before_fallback,
            scorer=SimpleHeuristicScorer(**cfg.heuristic),
            prompt_builder=SearchPromptBuilder(
                max_skeleton_feedbacks=cfg.max_skeleton_feedbacks,
                max_tactic_feedbacks=cfg.max_tactic_feedbacks,
            ),
        )

        for i, lean_path in enumerate(lean_files):
            print(f"[{i+1}/{len(lean_files)}] Rolling out: {lean_path}")
            
            with open(lean_path, encoding="utf-8") as f:
                content = f.read()
            
            vr = verifier.verify(content)
            sorries = vr.get("sorries") or []
            goal_str = (sorries[0] or {}).get("goal", "") if sorries else ""
            root_state = parse_proof_state(goal_str or "", header=DEFAULT_OPEN)
            
            if root_state is None:
                print(f"Warning: Could not parse theorem from: {lean_path}. Skipping.")
                continue

            # Determine output path
            if args.lean_dir:
                rel_path = os.path.relpath(lean_path, args.lean_dir)
                out_path = os.path.join(args.out_json, rel_path.replace(".lean", ".json"))
            else:
                out_path = args.out_json
                if out_path.endswith("/") or os.path.isdir(out_path):
                    out_path = os.path.join(out_path, os.path.basename(lean_path).replace(".lean", ".json"))

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            try:
                _, graph, q_values = rollout_engine.rollout(root_state)
                GraphLogger().save_json(graph, root_state, q_values, filepath=out_path)
                print(f"Saved to: {out_path}")
            except Exception as e:
                print(f"Error during rollout for {lean_path}: {e}")

    finally:
        if policy_server is not None:
            policy_server.kill()
        verifier.close()


if __name__ == "__main__":
    main()

# Example usage:
# python3 run_rollout.py --config configs/ --lean-dir problems/ --out-json outputs/ --policy
