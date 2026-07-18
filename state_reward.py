"""
state_reward.py — State vector construction, running normalisation,
                  and reward computation.

State vector (15-dim)
────────────────────────────────
 [0]  grad_norm_mean       — mean L2 norm of client gradients
 [1]  grad_norm_std        — std of L2 norms
 [2]  grad_cosine_mean     — mean pairwise cosine similarity
 [3]  grad_cosine_std      — std of pairwise cosine similarities
 [4]  grad_variance        — mean element-wise variance
 [5]  loss_delta           — change in global val loss
 [6]  acc_delta            — change in global val accuracy
 [7]  client_divergence    — variance of per-client val losses
 [8]  alpha_estimate       — MLE estimate of Dirichlet α
 [9]  round_normalized     — round / total_rounds
[10]  prev_defense_0       — one-hot: was last action 0?
[11]  prev_defense_1       — one-hot: was last action 1?
[12]  prev_defense_2       — one-hot: was last action 2?
[13]  prev_defense_3       — one-hot: was last action 3?
[14]  prev_defense_4       — one-hot: was last action 4?
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from config import (
    NUM_DEFENSES, FULL_ROUNDS, NORM_MOMENTUM,
    R_ACC, R_PRIVACY, R_COST, R_UTILITY,
    DEFENSE_COST, REWARD_NORM_MOMENTUM, DEVICE
)

STATE_DIM = 15


# ─────────────────────────────────────────────────────────────────────────────
# RUNNING MIN-MAX NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

class RunningNormaliser:
    """
    Online min-max normaliser with exponential moving statistics.
    Normalises each feature independently to [0,1].

    Uses (1 - NORM_MOMENTUM) as the update weight so that recent values
    have more influence than distant history (soft min/max).
    """

    def __init__(self, dim: int, momentum: float = NORM_MOMENTUM):
        self.dim      = dim
        self.momentum = momentum
        self._min = np.full(dim, np.inf,  dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        self._initialised = False

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        if not self._initialised:
            self._min = x.copy().astype(np.float64)
            self._max = x.copy().astype(np.float64)
            self._initialised = True
        else:
            alpha = 1.0 - self.momentum
            self._min = self.momentum * self._min + alpha * np.minimum(self._min, x)
            self._max = self.momentum * self._max + alpha * np.maximum(self._max, x)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Map x to [0,1] using running min/max."""
        if not self._initialised:
            return np.zeros_like(x, dtype=np.float32)
        denom = self._max - self._min
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        normed = (x - self._min) / denom
        return np.clip(normed, 0.0, 1.0).astype(np.float32)

    def update_and_normalise(self, x: np.ndarray) -> np.ndarray:
        self.update(x)
        return self.normalise(x)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_stats(
    gradients: Dict[int, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute gradient-level statistics from a dict of {cid: flat_gradient}.

    Returns dict with keys:
        grad_norm_mean, grad_norm_std,
        grad_cosine_mean, grad_cosine_std,
        grad_variance
    """
    cids = list(gradients.keys())
    if not cids:
        return {k: 0.0 for k in [
            "grad_norm_mean", "grad_norm_std",
            "grad_cosine_mean", "grad_cosine_std", "grad_variance"
        ]}

    mat   = torch.stack([gradients[c] for c in cids], dim=0)  # (N, D)
    norms = mat.norm(dim=1)                                     # (N,)

    norms_clamped  = torch.clamp(norms, max=1e6)
    grad_norm_mean = float(norms_clamped.mean().item())
    grad_norm_std  = float(norms_clamped.std().item()) if len(cids) > 1 else 0.0

    # Element-wise variance across clients
    mat_clamped   = torch.clamp(mat, min=-1e6, max=1e6)
    grad_variance = float(mat_clamped.var(dim=0).mean().item()) if len(cids) > 1 else 0.0

    # Pairwise cosine similarities
    if len(cids) < 2:
        grad_cosine_mean = 1.0
        grad_cosine_std  = 0.0
    else:
        # Normalise rows
        mat_n = mat / (norms.unsqueeze(1) + 1e-8)
        cos   = mat_n @ mat_n.T                                  # (N, N)
        # Extract upper triangle (excluding diagonal)
        idx = torch.triu_indices(len(cids), len(cids), offset=1)
        cos_vals = cos[idx[0], idx[1]]
        grad_cosine_mean = float(cos_vals.mean().item())
        grad_cosine_std  = float(cos_vals.std().item()) if cos_vals.numel() > 1 else 0.0

    return {
        "grad_norm_mean":   grad_norm_mean,
        "grad_norm_std":    grad_norm_std,
        "grad_cosine_mean": grad_cosine_mean,
        "grad_cosine_std":  grad_cosine_std,
        "grad_variance":    grad_variance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE VECTOR ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_state_vector(
    gradients: Dict[int, torch.Tensor],
    prev_loss: float,
    curr_loss: float,
    prev_acc: float,
    curr_acc: float,
    client_val_losses: Dict[int, float],
    alpha_estimate: float,
    round_idx: int,
    total_rounds: int,
    prev_defense: int,
) -> np.ndarray:
    """
    Assemble the raw (un-normalised) 15-dimensional state vector.

    Returns np.ndarray of shape (15,), dtype float64.
    """
    stats = compute_gradient_stats(gradients)

    loss_delta = curr_loss - prev_loss
    acc_delta  = curr_acc  - prev_acc

    # Client divergence: variance of per-client validation losses
    val_loss_vals = list(client_val_losses.values())
    client_divergence = float(np.var(val_loss_vals)) if len(val_loss_vals) > 1 else 0.0

    round_norm = round_idx / max(total_rounds - 1, 1)

    # One-hot encode previous defense
    prev_def_onehot = np.zeros(NUM_DEFENSES, dtype=np.float64)
    if 0 <= prev_defense < NUM_DEFENSES:
        prev_def_onehot[prev_defense] = 1.0

    def _safe(v):
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return float(v)

    raw = np.array([
        _safe(stats["grad_norm_mean"]),
        _safe(stats["grad_norm_std"]),
        _safe(stats["grad_cosine_mean"]),
        _safe(stats["grad_cosine_std"]),
        _safe(stats["grad_variance"]),
        _safe(loss_delta),
        _safe(acc_delta),
        _safe(client_divergence),
        _safe(alpha_estimate),
        _safe(round_norm),
    ], dtype=np.float64)

    return np.concatenate([raw, prev_def_onehot])   # shape (15,)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION UTILITIES (accuracy, ASR)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """Return top-1 accuracy on a data loader."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(
    model: torch.nn.Module,
    trigger_loader: torch.utils.data.DataLoader,
    target_class: int,
    device: torch.device = DEVICE,
) -> float:
    """
    Attack Success Rate: fraction of triggered test inputs classified as
    `target_class`.
    """
    model.eval()
    success, total = 0, 0
    for imgs, _ in trigger_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1)
        success += (preds == target_class).sum().item()
        total   += imgs.size(0)
    return success / max(total, 1)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """Return mean CrossEntropyLoss on a data loader."""
    import torch.nn as nn
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, steps = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        total  += criterion(model(imgs), labels).item()
        steps  += 1
    return total / max(steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# REWARD COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

class RewardNormaliser:
    """Running mean ± std normaliser for reward values (momentum-based)."""

    def __init__(self, momentum: float = REWARD_NORM_MOMENTUM):
        self.momentum = momentum
        self._mean = 0.0
        self._var  = 1.0
        self._count = 0

    def update_and_normalise(self, r: float) -> float:
        alpha = 1.0 - self.momentum
        self._count += 1
        self._mean = self.momentum * self._mean + alpha * r
        self._var  = self.momentum * self._var  + alpha * (r - self._mean) ** 2
        std = max(self._var ** 0.5, 1e-8)
        return (r - self._mean) / std


def compute_reward(action, prev_acc, curr_acc, privacy_score,
                   acc_before_defense, acc_after_defense):

    delta_acc    = float(np.clip(curr_acc - prev_acc, -1.0, 1.0))
    cost_penalty = DEFENSE_COST.get(action, 0.0)
    utility_loss = max(0.0, acc_before_defense - acc_after_defense)

    # Use smoothed accuracy signal — reward based on level not delta
    # This is more stable for PPO in Non-IID environments
    acc_level = float(np.clip(curr_acc, 0.0, 1.0))

    # FIX (see project changelog): these weights were hardcoded here and
    # did NOT reference config.R_ACC / R_PRIVACY / R_COST / R_UTILITY,
    # despite config.py's own docstring saying "never hardcode values
    # elsewhere". Sweeping the config weights for a sensitivity ablation
    # previously had zero effect on training. Now wired through properly.
    # Note the acc term is split 50/50 between delta and level, both
    # under the single R_ACC weight, matching the original 0.25+0.25 split.
    reward = (
        (R_ACC / 2.0) * delta_acc     # change in accuracy
      + (R_ACC / 2.0) * acc_level     # absolute accuracy level
      + R_PRIVACY      * privacy_score  # privacy protection
      - R_COST          * cost_penalty  # computational cost
      - R_UTILITY       * utility_loss  # defense utility loss
    )
    return float(reward)