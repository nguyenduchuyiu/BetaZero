import argparse
import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from betazero.core.nodes import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.search.graph.and_or_graph import ANDORGraph
from betazero.search.rollout.levelwise_rollout import LevelwiseRollout
from betazero.search.sorrifier import Sorrifier
from betazero.search.reward import RewardCalculator
from betazero.utils.config import Config
from betazero.policy.vllm_server import VLLMServer
from betazero.policy.api_server import APIServer
from betazero.utils.lean_parse import parse_proof_state
from betazero.utils.lean_cmd import DEFAULT_OPEN
from betazero.utils.graph_logger import GraphLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--lean-file", type=str, default="problems/aime_1984_p5.lean")
    ap.add_argument("--out-json", type=str, default="outputs/rollout_aime_1984_p5.json")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--lean-timeout", type=int, default=None)
    ap.add_argument("--policy", type=str, choices=["vllm", "api"], default="vllm")
    args = ap.parse_args()

    cfg: Config = Config()
    if args.config:
        cfg = Config.from_yaml(args.config)

    lean_path = os.path.join(os.path.dirname(__file__), args.lean_file)
    if not os.path.exists(lean_path):
        raise FileNotFoundError(lean_path)

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

    with open(lean_path, encoding="utf-8") as f:
        content = f.read()
    vr = verifier.verify(content)
    sorries = vr.get("sorries") or []
    goal_str = (sorries[0] or {}).get("goal", "") if sorries else ""
    root_state = parse_proof_state(goal_str or "", header=DEFAULT_OPEN)
    
    if root_state is None:
        raise RuntimeError(f"Could not parse theorem from: {lean_path}")

    policy_server = None
    try:
        # Use cfg loaded from --config for model/vLLM/API settings.
        if args.policy == "vllm":
            policy_server = VLLMServer(cfg)
        else:
            policy_server = APIServer(cfg)
            
        policy_server.start(args.adapter)
        policy = policy_server

        rollout = LevelwiseRollout(
            policy=policy,
            lean=lean,
            sorrifier=sorrifier,
            reward=reward,
            K=K,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        _, _, graph, q_values = rollout.rollout(root_state)

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        GraphLogger().save_json(graph, root_state, q_values, filepath=args.out_json)

    finally:
        if policy_server is not None:
            policy_server.kill()
        verifier.close()


if __name__ == "__main__":
    main()

# python3 run_rollout.py --config configs/qwen2.5_0.5b.yaml --lean-file problems/aime_1984_p5.lean --out-json outputs/rollout_aime_1984_p5.json --policy api
