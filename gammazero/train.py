from __future__ import annotations

import os
import gc
import sys

# Giảm phân mảnh bộ nhớ (khuyến nghị cho CUDA OOM)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gammazero.utils.config import Config
from gammazero.utils.dataloader import TheoremDataset
from gammazero.utils.graph_logger import GraphLogger
from gammazero.utils.logger import setup as setup_logger
from gammazero.policy.vllm_server import VLLMServer
from gammazero.policy.trainable_policy import TrainablePolicy
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search import Sorrifier
from gammazero.search import RewardCalculator
from gammazero.search import BestFirstRollout
from gammazero.search import GRPOTrainer


def train(cfg: Config = Config()):
    log_dir  = cfg.run_log_dir
    ckpt_dir = cfg.run_checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, writer = setup_logger(log_dir)
    cfg.save(log_dir)
    logger.info(f"Run: {cfg.run_name}  |  {cfg}")

    scheduler = Lean4ServerScheduler(max_concurrent_requests=cfg.lean_workers,
                                     timeout=cfg.lean_timeout, name=cfg.run_name)
    lean      = LeanEnv(scheduler)
    sorrifier = Sorrifier(scheduler)
    reward    = RewardCalculator()
    dataset   = TheoremDataset(cfg.dataset_dir, scheduler=scheduler)
    trainer   = GRPOTrainer(
        lr=cfg.lr, eps_clip=cfg.eps_clip, beta_kl=cfg.beta_kl,
        grpo_epochs=cfg.grpo_epochs, mini_batch_size=cfg.mini_batch_size,
        accumulation_steps=cfg.grpo_accumulation_steps,
    )
    vllm      = VLLMServer(cfg)

    adapter_path: str | None = None  # updated each iteration
    grpo_buffer: list = []

    try:
        for iteration in tqdm(range(1, cfg.total_iterations + 1), desc=cfg.run_name):
            last_iter = iteration == cfg.total_iterations
            theorems = dataset.sample(cfg.theorems_per_iter)

            samples = []
            try:
                # ── Phase 1: Rollout with vLLM subprocess ─────────────────────
                vllm.start(adapter_path)
                for j, thm in enumerate(theorems):
                    rollout = BestFirstRollout(
                        vllm, lean, sorrifier, reward,
                        max_depth=cfg.max_depth,
                        max_nodes=cfg.max_nodes,
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
                    )
                    batch, g, qv = rollout.rollout(thm)
                    samples.extend(batch)
                    if cfg.rollout_graph_log_dir:
                        path = os.path.join(
                            cfg.rollout_graph_log_dir, cfg.run_name, f"iter{iteration:04d}_thm{j:02d}.json"
                        )
                        GraphLogger().save_json(g, thm, qv, filepath=path)
            finally:
                vllm.kill()
                gc.collect()
                torch.cuda.empty_cache()

            if cfg.min_samples_for_grpo > 0:
                grpo_buffer.extend(samples)
                train_batch = grpo_buffer
                need = cfg.min_samples_for_grpo
                do_train = len(train_batch) >= need or (last_iter and len(train_batch) > 0)
            else:
                train_batch = samples
                need = 1
                do_train = len(train_batch) > 0

            logger.info(
                f"[{iteration:4d}] rollout: {len(samples)} samples  "
                f"buffer={len(grpo_buffer) if cfg.min_samples_for_grpo > 0 else len(samples)}"
                f"{'/' + str(need) if cfg.min_samples_for_grpo > 0 else ''}"
            )

            m = {
                "loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
                "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0,
            }
            if do_train:
                policy = TrainablePolicy(cfg, adapter_path)
                try:
                    m = trainer.update(policy, train_batch)
                    adapter_path = os.path.join(ckpt_dir, f"iter{iteration:04d}")
                    policy.save(adapter_path)
                finally:
                    policy.unload()
                    del policy
                    gc.collect()
                    torch.cuda.empty_cache()
                if cfg.min_samples_for_grpo > 0:
                    grpo_buffer.clear()
            elif cfg.min_samples_for_grpo > 0:
                logger.info(
                    f"[{iteration:4d}] skip GRPO: buffer {len(grpo_buffer)}/{cfg.min_samples_for_grpo}"
                )

            logger.info(
                f"[{iteration:4d}]  loss={m['loss']:.4f}  kl={m['kl']:.4f}  "
                f"r_env={m['r_env_mean']:.3f}  Q={m['Q_mean']:.3f}  "
                f"solve={m['solve_rate']:.2%}  samples={m['n_samples']}  groups={m['n_groups']}"
            )
            writer.add_scalar("train/loss",       m["loss"],       iteration)
            writer.add_scalar("train/kl",         m["kl"],         iteration)
            writer.add_scalar("train/r_env_mean", m["r_env_mean"], iteration)
            writer.add_scalar("train/Q_mean",     m["Q_mean"],     iteration)
            writer.add_scalar("train/solve_rate", m["solve_rate"], iteration)
            writer.add_scalar("train/n_samples",  m["n_samples"],  iteration)

            if do_train and iteration % cfg.checkpoint_every == 0:
                logger.info(f"Checkpoint at iter {iteration}: {adapter_path}")
    finally:
        writer.close()
        scheduler.close()

if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "configs/deepseek_r1_distill_7B.yaml"
    if not os.path.exists(yaml_path) and os.path.exists(os.path.join("configs", yaml_path)):
        yaml_path = os.path.join("configs", yaml_path)
    train(Config.from_yaml(yaml_path))
