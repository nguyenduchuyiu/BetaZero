import os
import gc
import shutil
import subprocess
import torch
from tqdm import tqdm

from betazero.utils.config import Config
from betazero.utils.dataloader import TheoremDataset
from betazero.utils.logger import setup as setup_logger
from betazero.policy.vllm_server import VLLMServer
from betazero.policy.trainable_policy import TrainablePolicy
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.search import Sorrifier
from betazero.search import RewardCalculator
from betazero.search import LevelwiseRollout
from betazero.search import GRPOTrainer


def _log_vram(logger, tag: str):
    if not torch.cuda.is_available():
        logger.info(f"[vram] {tag}: cuda not available")
        return

    free_b, total_b = torch.cuda.mem_get_info()
    logger.info(f"[vram] {tag}: free={free_b / 1024**3:.2f}GiB total={total_b / 1024**3:.2f}GiB")

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=2,
            ).strip()
            logger.info(f"[vram] {tag}: procs:\n{out if out else '(none)'}")
        except Exception as e:
            logger.info(f"[vram] {tag}: nvidia-smi failed: {e}")


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
    trainer   = GRPOTrainer(lr=cfg.lr, eps_clip=cfg.eps_clip, beta_kl=cfg.beta_kl,
                            grpo_epochs=cfg.grpo_epochs, mini_batch_size=cfg.mini_batch_size)
    vllm      = VLLMServer(cfg)

    adapter_path: str | None = None  # updated each iteration

    try:
        for iteration in tqdm(range(1, cfg.total_iterations + 1), desc=cfg.run_name):
            theorems = dataset.sample(cfg.theorems_per_iter)

            samples = []
            try:
                # ── Phase 1: Rollout with vLLM subprocess ─────────────────────
                vllm.start(adapter_path)
                rollout = LevelwiseRollout(
                    vllm, lean, sorrifier, reward,
                    K=cfg.K, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes
                )
                for thm in theorems:
                    samples.extend(rollout.rollout(thm))
            finally:
                vllm.kill()
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f"[{iteration:4d}] rollout: {len(samples)} samples")
            _log_vram(logger, f"iter{iteration:04d} pre-train")

            # ── Phase 2: GRPO update with PyTorch ─────────────────────────
            policy = TrainablePolicy(cfg, adapter_path)
            try:
                m = trainer.update(policy, samples)
                adapter_path = os.path.join(ckpt_dir, f"iter{iteration:04d}")
                policy.save(adapter_path)
            finally:
                policy.unload()
                del policy
                gc.collect()
                torch.cuda.empty_cache()
                _log_vram(logger, f"iter{iteration:04d} post-train-cleanup")

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

            if iteration % cfg.checkpoint_every == 0:
                logger.info(f"Checkpoint at iter {iteration}: {adapter_path}")
    finally:
        writer.close()
        scheduler.close()
