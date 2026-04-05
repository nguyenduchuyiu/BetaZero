from __future__ import annotations
import random
from collections import defaultdict

import torch

from betazero.core import ProofState, Action
from betazero.policy import TrainablePolicy

_EMPTY = {"loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
          "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0}


class GRPOTrainer:
    """Applies GRPO updates to a policy from rollout samples collected during search."""

    def __init__(self, lr: float = 1e-5, eps_clip: float = 0.2, beta_kl: float = 0.01,
                 grpo_epochs: int = 1, mini_batch_size: int = 8, accumulation_steps: int = 1):
        self.lr                 = lr
        self.eps_clip           = eps_clip
        self.beta_kl            = beta_kl
        self.grpo_epochs        = grpo_epochs
        self.mini_batch_size    = mini_batch_size
        self.accumulation_steps = max(1, accumulation_steps)

    def update(self, policy: TrainablePolicy,
               samples: list[tuple[ProofState, Action, float, float]],
               aux_samples: list[tuple[ProofState, Action, float, float]] | None = None) -> dict:
        """GRPO update on pre-collected rollout samples. Returns metrics dict."""
        if not samples:
            return _EMPTY

        # Gom nhóm toàn bộ sample theo prompt
        raw_groups: dict[str, list[int]] = defaultdict(list)
        for i, (_, action, _, _) in enumerate(samples):
            raw_groups[action.prompt].append(i)

        adv  = [0.0] * len(samples)
        keep = [False] * len(samples)
        
        # Chỉ lấy các nhóm đủ điều kiện làm GRPO (size >= 2 và std > 0)
        valid_prompt_groups: dict[str, list[int]] = {}
        
        for p_text, idxs in raw_groups.items():
            returns = [
                samples[i][2] if samples[i][1].action_type == "tactic" else samples[i][3]
                for i in idxs
            ]
            mean_r  = sum(returns) / len(returns)
            std_r   = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
            
            if std_r < 1e-8 or len(idxs) < 2:
                continue
                
            group_keep_idxs = []
            for i, r in zip(idxs, returns):
                adv[i] = (r - mean_r) / std_r
                keep[i] = True
                group_keep_idxs.append(i)
                
            valid_prompt_groups[p_text] = group_keep_idxs

        train_idx = [i for i in range(len(samples)) if keep[i]]
        if not train_idx:
            return _EMPTY

        # Bóc tách dữ liệu sạch để train
        # Cần map index cũ -> index mới trong danh sách train_idx
        old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate(train_idx)}
        train_groups = {
            p_text: [old_to_new_idx[i] for i in idxs]
            for p_text, idxs in valid_prompt_groups.items()
        }

        states  = [samples[i][0] for i in train_idx]
        actions = [samples[i][1].content for i in train_idx]
        prompts = [samples[i][1].prompt for i in train_idx]
        device  = next(policy.parameters()).device
        adv_t   = torch.tensor([adv[i] for i in train_idx], dtype=torch.float32, device=device)

        # Tính Reference Log-probs
        with torch.no_grad():
            ref_lp = self._batch_lp(policy, states, actions, prompts, train_groups, disable_adapter=True)

        optimizer  = torch.optim.Adam(policy.parameters(), lr=self.lr)
        total_loss = total_kl = 0.0
        n_steps    = 0
        acc        = self.accumulation_steps

        for _ in range(self.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            
            # Tạo mini-batches sao cho bảo toàn cụm prompt
            mb_list = self._minibatches(train_groups)
            n_mb = len(mb_list)
            
            for mb_idx, mb in enumerate(mb_list):
                mb_s = [states[j] for j in mb]
                mb_a = [actions[j] for j in mb]
                mb_p = [prompts[j] for j in mb]
                mb_adv = adv_t[mb]
                mb_ref = ref_lp[mb]

                mb_theta = policy.log_probs(mb_s, mb_a, mb_p)
                ratio    = torch.exp(mb_theta - mb_ref)
                surr     = torch.min(ratio * mb_adv,
                                     torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv)
                kl   = ratio - (mb_theta - mb_ref) - 1.0
                loss = -(surr - self.beta_kl * kl).mean() / acc

                loss.backward()
                accum_counter += 1
                is_last = mb_idx + 1 == n_mb
                
                if accum_counter >= acc or is_last:
                    if is_last and accum_counter < acc:
                        scale = acc / accum_counter
                        for p in policy.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                total_loss += loss.item() * acc
                total_kl   += kl.mean().item()
                n_steps    += 1

        r_envs   = [s[2] for s in samples]
        qs       = [s[3] for s in samples]
        return {
            "loss":       total_loss / n_steps,
            "kl":         total_kl   / n_steps,
            "n_samples":  len(train_idx),
            "n_groups":   len(valid_prompt_groups),
            "r_env_mean": sum(r_envs) / len(r_envs),
            "Q_mean":     sum(qs)     / len(qs),
            "solve_rate": sum(1 for s in samples if s[3] > 0) / len(samples),
        }

    def _minibatches(self, groups_dict: dict[str, list[int]]):
        """
        Chia mini-batch sao cho các action chung prompt được đóng gói cùng nhau.
        Tối đa hóa hiệu năng của Shared Prefill trong TrainablePolicy.
        """
        all_mbs = []
        prompts = list(groups_dict.keys())
        random.shuffle(prompts)
        
        for p in prompts:
            idxs = groups_dict[p].copy()
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.mini_batch_size):
                all_mbs.append(idxs[i:i + self.mini_batch_size])
                
        random.shuffle(all_mbs)
        return all_mbs

    def _batch_lp(self, policy, states, actions, prompts, groups_dict,
                  disable_adapter: bool = False) -> torch.Tensor:
        """Dùng logic minibatches gom nhóm để tính reference log probs tốc độ cao."""
        results = torch.zeros(len(states), dtype=torch.float32, device=next(policy.parameters()).device)
        
        mb_list = self._minibatches(groups_dict)
        for mb in mb_list:
            s = [states[i] for i in mb]
            a = [actions[i] for i in mb]
            p = [prompts[i] for i in mb]
            
            mb_scores = policy.log_probs(s, a, p, disable_adapter=disable_adapter)
            for i, score in zip(mb, mb_scores):
                results[i] = score
                
        return results