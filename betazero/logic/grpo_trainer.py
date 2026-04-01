from __future__ import annotations
from collections import defaultdict

import torch

from betazero.data.nodes import ProofState, Action
from betazero.model.trainable_policy import TrainablePolicy

_EMPTY = {"loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
          "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0}


class GRPOTrainer:
    """GRPO update step. Stateless except hyperparams — call update() each iteration."""

    def __init__(self, lr: float = 1e-5, eps_clip: float = 0.2, beta_kl: float = 0.01,
                 grpo_epochs: int = 1, mini_batch_size: int = 8):
        self.lr              = lr
        self.eps_clip        = eps_clip
        self.beta_kl         = beta_kl
        self.grpo_epochs     = grpo_epochs
        self.mini_batch_size = mini_batch_size

    def update(self, policy: TrainablePolicy,
               samples: list[tuple[ProofState, Action, float, float]]) -> dict:
        """GRPO update on pre-collected rollout samples. Returns metrics dict."""
        if not samples:
            return _EMPTY

        # Group by (context, goal, action_type); tactics use r_env, skeletons use Q
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, (state, action, _, _) in enumerate(samples):
            groups[(state.context, state.goal, action.action_type)].append(i)

        adv  = [0.0] * len(samples)
        keep = [False] * len(samples)
        for (_, _, atype), idxs in groups.items():
            returns = [samples[i][2] if atype == "tactic" else samples[i][3] for i in idxs]
            mean_r  = sum(returns) / len(returns)
            std_r   = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_r < 1e-8:
                continue
            for i, r in zip(idxs, returns):
                adv[i], keep[i] = (r - mean_r) / std_r, True

        train_idx = [i for i in range(len(samples)) if keep[i]]
        if not train_idx:
            return _EMPTY

        states       = [samples[i][0] for i in train_idx]
        actions      = [samples[i][1].content for i in train_idx]
        action_types = [samples[i][1].action_type for i in train_idx]
        device       = next(policy.parameters()).device
        adv_t        = torch.tensor([adv[i] for i in train_idx], dtype=torch.float32, device=device)

        # Reference log-probs: base model (LoRA disabled) — no separate ref model needed
        with torch.no_grad():
            ref_lp = self._batch_lp(policy, states, actions, action_types, disable_adapter=True)

        optimizer  = torch.optim.Adam(policy.parameters(), lr=self.lr)
        total_loss = total_kl = 0.0
        n_steps    = 0

        for _ in range(self.grpo_epochs):
            for mb in self._minibatches(len(states)):
                mb_s   = [states[j] for j in mb]
                mb_a   = [actions[j] for j in mb]
                mb_at  = [action_types[j] for j in mb]
                mb_adv = adv_t[mb]
                mb_ref = ref_lp[mb]

                mb_theta = policy.log_probs(mb_s, mb_a, mb_at)
                ratio    = torch.exp(mb_theta - mb_ref)
                surr     = torch.min(ratio * mb_adv,
                                     torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv)
                kl   = ratio - (mb_theta - mb_ref) - 1.0  # non-linear KL: r - log(r) - 1
                loss = -(surr - self.beta_kl * kl).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_kl   += kl.mean().item()
                n_steps    += 1

        r_envs   = [s[2] for s in samples]
        qs       = [s[3] for s in samples]
        n_groups = sum(1 for idxs in groups.values() if any(keep[i] for i in idxs))
        return {
            "loss":       total_loss / n_steps,
            "kl":         total_kl   / n_steps,
            "n_samples":  len(train_idx),
            "n_groups":   n_groups,
            "r_env_mean": sum(r_envs) / len(r_envs),
            "Q_mean":     sum(qs)     / len(qs),
            "solve_rate": sum(1 for s in samples if s[3] > 0) / len(samples),
        }

    def _minibatches(self, n: int):
        indices = torch.randperm(n).tolist()
        for i in range(0, n, self.mini_batch_size):
            yield indices[i:i + self.mini_batch_size]

    def _batch_lp(self, policy, states, actions, action_types,
                  disable_adapter: bool = False) -> torch.Tensor:
        results = []
        for i in range(0, len(states), self.mini_batch_size):
            s  = states[i:i + self.mini_batch_size]
            a  = actions[i:i + self.mini_batch_size]
            at = action_types[i:i + self.mini_batch_size]
            results.append(policy.log_probs(s, a, at, disable_adapter=disable_adapter))
        return torch.cat(results)
