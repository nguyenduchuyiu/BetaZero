from __future__ import annotations
from collections import defaultdict

import torch
from contextlib import nullcontext

from nodes import ProofState, Action
from rollout import LevelwiseRollout, PolicyModel


class GRPOTrainer:
    """GRPO training loop with decoupled tactic/skeleton groups (Section 7)."""

    def __init__(self, policy: PolicyModel, rollout: LevelwiseRollout,
                 lr: float = 1e-5, eps_clip: float = 0.2, beta_kl: float = 0.01,
                 grpo_epochs: int = 4, mini_batch_size: int = 8):
        self.policy = policy
        self.rollout = rollout
        self.eps_clip = eps_clip
        self.beta_kl = beta_kl
        self.grpo_epochs = grpo_epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, theorems: list[ProofState]) -> dict:
        """One training iteration: rollout all theorems, then GRPO update."""
        samples: list[tuple[ProofState, Action, float, float]] = []
        for thm in theorems:
            samples.extend(self.rollout.rollout(thm))
        if not samples:
            return {"loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
                    "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0}

        # Group by (context, goal, action_type); tactics use r_env, skeletons use Q (Sec 7.1/7.2)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, (state, action, _, _) in enumerate(samples):
            groups[(state.context, state.goal, action.action_type)].append(i)

        adv  = [0.0] * len(samples)
        keep = [False] * len(samples)
        for (_, _, atype), idxs in groups.items():
            returns = [samples[i][2] if atype == "tactic" else samples[i][3] for i in idxs]
            mean_r = sum(returns) / len(returns)
            std_r  = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_r < 1e-8:
                continue  # Skip degenerate groups (Sec 7.1)
            for i, r in zip(idxs, returns):
                adv[i]  = (r - mean_r) / std_r
                keep[i] = True

        train_idx = [i for i in range(len(samples)) if keep[i]]
        if not train_idx:
            return {"loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
                    "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0}

        states       = [samples[i][0] for i in train_idx]
        actions      = [samples[i][1].content for i in train_idx]
        action_types = [samples[i][1].action_type for i in train_idx]
        device       = next(self.policy.parameters()).device
        adv_t        = torch.tensor([adv[i] for i in train_idx], dtype=torch.float32, device=device)

        # Compute reference log-probs once in mini-batches to avoid OOM
        ref_log_probs = self._batch_log_probs(
            self.policy, states, 
            actions, action_types, 
            no_grad=True, disable_adapter=True)

        total_loss = total_kl = 0.0
        n_steps = 0
        for _ in range(self.grpo_epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.mini_batch_size):
                mb = indices[i:i + self.mini_batch_size]
                mb_s   = [states[j.item()] for j in mb]
                mb_a   = [actions[j.item()] for j in mb]
                mb_at  = [action_types[j.item()] for j in mb]
                mb_adv = adv_t[mb]
                mb_ref = ref_log_probs[mb]

                mb_theta = self.policy.log_probs(mb_s, mb_a, mb_at)
                ratio = torch.exp(mb_theta - mb_ref)
                surr  = torch.min(ratio * mb_adv,
                                  torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv)
                # Non-linear KL approximation: r - log(r) - 1
                kl   = ratio - (mb_theta - mb_ref) - 1.0
                loss = -(surr - self.beta_kl * kl).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_kl   += kl.mean().item()
                n_steps    += 1

        r_envs = [s[2] for s in samples]
        qs     = [s[3] for s in samples]
        solved = sum(1 for s in samples if s[3] > 0.0)
        n_groups = sum(1 for idxs in groups.values() if any(keep[i] for i in idxs))
        return {
            "loss":      total_loss / n_steps if n_steps > 0 else 0.0,
            "kl":        total_kl   / n_steps if n_steps > 0 else 0.0,
            "n_samples": len(train_idx),
            "n_groups":  n_groups,
            "r_env_mean": sum(r_envs) / len(r_envs),
            "Q_mean":     sum(qs)     / len(qs),
            "solve_rate": solved / len(samples),
        }

    def _batch_log_probs(self, model: PolicyModel, states, actions, action_types,
                         no_grad: bool = False, disable_adapter: bool = False) -> torch.Tensor:
        """Compute log_probs in mini-batches; optionally disable gradients."""
        results = []
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        adapter_ctx = model.model.disable_adapter() if disable_adapter else nullcontext()
        with ctx, adapter_ctx:
            for i in range(0, len(states), self.mini_batch_size):
                s  = states[i:i + self.mini_batch_size]
                a  = actions[i:i + self.mini_batch_size]
                at = action_types[i:i + self.mini_batch_size]
                results.append(model.log_probs(s, a, at))
        return torch.cat(results)
