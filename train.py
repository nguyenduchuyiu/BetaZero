import os
import sys
import torch
from tqdm import tqdm

from config import Config
from dataloader import TheoremDataset
from logger import setup as setup_logger
from policy_model import PolicyModel
from lean_env import LeanEnv
from lean_verifier import Lean4ServerScheduler
from sorrifier import Sorrifier
from reward import RewardCalculator
from rollout import LevelwiseRollout
from grpo_trainer import GRPOTrainer


def train(cfg: Config = Config()):
    log_dir  = cfg.run_log_dir
    ckpt_dir = cfg.run_checkpoint_dir

    logger, writer = setup_logger(log_dir)
    cfg.save(log_dir)  # snapshot config alongside logs
    logger.info(f"Run: {cfg.run_name}  |  {cfg}")

    logger.info(f"Loading policy: {cfg.model_name}")
    policy     = PolicyModel(cfg)

    scheduler = Lean4ServerScheduler(max_concurrent_requests=cfg.lean_workers,
                                     timeout=cfg.lean_timeout, name=cfg.run_name)
    lean      = LeanEnv(scheduler)
    sorrifier = Sorrifier(scheduler)
    reward    = RewardCalculator()
    rollout   = LevelwiseRollout(policy, lean, sorrifier, reward,
                                 K=cfg.K, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes)
    trainer   = GRPOTrainer(policy, rollout,
                            lr=cfg.lr, eps_clip=cfg.eps_clip, beta_kl=cfg.beta_kl,
                            grpo_epochs=cfg.grpo_epochs,
                            mini_batch_size=cfg.mini_batch_size)
    dataset   = TheoremDataset(cfg.dataset_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    for iteration in tqdm(range(1, cfg.total_iterations + 1), desc=cfg.run_name):
        theorems = dataset.sample(cfg.theorems_per_iter)
        m        = trainer.train(theorems)

        logger.info(
            f"[{iteration:4d}]  loss={m['loss']:.4f}  kl={m['kl']:.4f}  "
            f"r_env={m['r_env_mean']:.3f}  Q={m['Q_mean']:.3f}  "
            f"solve={m['solve_rate']:.2%}  "
            f"samples={m['n_samples']}  groups={m['n_groups']}"
        )
        writer.add_scalar("train/loss",       m["loss"],       iteration)
        writer.add_scalar("train/kl",         m["kl"],         iteration)
        writer.add_scalar("train/r_env_mean", m["r_env_mean"], iteration)
        writer.add_scalar("train/Q_mean",     m["Q_mean"],     iteration)
        writer.add_scalar("train/solve_rate", m["solve_rate"], iteration)
        writer.add_scalar("train/n_samples",  m["n_samples"],  iteration)

        if iteration % cfg.checkpoint_every == 0:
            policy.model.save_pretrained(ckpt_dir)
            logger.info(f"LoRA Checkpoint saved at: {ckpt_dir}")

    writer.close()
    scheduler.close()


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(Config.from_yaml(yaml_path))
