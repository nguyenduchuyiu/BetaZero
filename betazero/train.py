from __future__ import annotations

import os
import gc
import torch
from tqdm import tqdm

from betazero.utils.config import Config
from betazero.utils.dataloader import TheoremDataset
from betazero.utils.graph_logger import GraphLogger
from betazero.utils.logger import setup as setup_logger
from betazero.policy.vllm_server import VLLMServer
from betazero.policy.trainable_policy import TrainablePolicy
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.search import Sorrifier
from betazero.search import RewardCalculator
from betazero.search import LevelwiseRollout
from betazero.search import GRPOTrainer


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
    dataset   = TheoremDataset(cfg.dataset_dir)
    trainer   = GRPOTrainer(
        lr=cfg.lr, eps_clip=cfg.eps_clip, beta_kl=cfg.beta_kl,
        grpo_epochs=cfg.grpo_epochs, mini_batch_size=cfg.mini_batch_size,
        accumulation_steps=cfg.grpo_accumulation_steps,
    )
    vllm      = VLLMServer(cfg)

    adapter_path: str | None = None  # updated each iteration
    grpo_buffer: list = []
    self_correction_buffer: list = []

    try:
        for iteration in tqdm(range(1, cfg.total_iterations + 1), desc=cfg.run_name):
            last_iter = iteration == cfg.total_iterations
            theorems = dataset.sample(cfg.theorems_per_iter)

            samples = []
            self_correction_samples: list = []
            try:
                # ── Phase 1: Rollout with vLLM subprocess ─────────────────────
                vllm.start(adapter_path)
                for j, thm in enumerate(theorems):
                    rollout = LevelwiseRollout(
                        vllm, lean, sorrifier, reward,
                        K=cfg.K, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes,
                    )
                    batch, g, qv = rollout.rollout(thm)
                    samples.extend(batch)
                    self_correction_samples.extend(rollout.self_correction_buffer)
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
                self_correction_buffer.extend(self_correction_samples)
                train_batch = grpo_buffer
                aux_train_batch = self_correction_buffer
                need = cfg.min_samples_for_grpo
                do_train = len(train_batch) >= need or (last_iter and len(train_batch) > 0)
            else:
                train_batch = samples
                aux_train_batch = self_correction_samples
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
                    m = trainer.update(policy, train_batch, aux_train_batch)
                    adapter_path = os.path.join(ckpt_dir, f"iter{iteration:04d}")
                    policy.save(adapter_path)
                finally:
                    policy.unload()
                    del policy
                    gc.collect()
                    torch.cuda.empty_cache()
                if cfg.min_samples_for_grpo > 0:
                    grpo_buffer.clear()
                    self_correction_buffer.clear()
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
