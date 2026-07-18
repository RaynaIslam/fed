"""
defenses.py — All 5 defense mechanisms for the FL defense pool.

Action 0: DP-SGD      — Gaussian mechanism on aggregated gradient
Action 1: FLTrust     — Server-gradient-cosine-weighted aggregation
Action 2: Multi-Krum  — Byzantine-robust selection via nearest-neighbor score
Action 3: FLAME       — Cosine-distance clustering + adaptive noise
Action 4: NoDefense   — Plain FedAvg (no modification)

All defenses accept:
    gradients : Dict[int, torch.Tensor]  — {client_id: flat gradient vector}
    global_model : nn.Module             — current global model (read-only)
    **kwargs                             — defense-specific extras

All defenses return:
    aggregated_gradient : torch.Tensor   — flat gradient to apply to global model
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    DP_NOISE_MULTIPLIER, DP_CLIP_NORM,
    KRUM_F,
    FLAME_MIN_CLUSTER,
    NUM_DEFENSES, DEVICE
)

 

def _stack(gradients: Dict[int, torch.Tensor]) -> Tuple[List[int], torch.Tensor]:
    """Return (ordered_cids, matrix of shape [n_clients, param_dim])."""
    cids = list(gradients.keys())
    mat  = torch.stack([gradients[c] for c in cids], dim=0)   # (N, D)
    return cids, mat


def _simple_avg(gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Unweighted mean of gradient vectors."""
    _, mat = _stack(gradients)
    return mat.mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION 4 — NoDefense (plain FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def defense_no_defense(
    gradients: Dict[int, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """Standard FedAvg: unweighted mean of all client gradients."""
    return _simple_avg(gradients)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION 0 — DP-SGD
# ─────────────────────────────────────────────────────────────────────────────
def _dp_clip_and_noise(
    mat_or_vec: torch.Tensor,
    noise_multiplier: float,
    clip_norm: float,
    n_for_noise_scale: int,
) -> torch.Tensor:
    """
    Shared clip+noise primitive used both for the server-side aggregate
    (defense_dp_sgd) and for a single client's gradient when we need to
    simulate "what does an eavesdropper see for THIS client if DP-SGD is
    the active defense" (used by gia.py for privacy evaluation).

    n_for_noise_scale: divisor used in the noise-multiplier scaling.
        - server-side aggregate call: pass N (number of clients averaged)
        - single-client call: pass 1 (no averaging happened yet)
    """
    D = mat_or_vec.shape[-1]
    sigma = noise_multiplier * clip_norm / math.sqrt(max(n_for_noise_scale, 1))
    noise = torch.randn_like(mat_or_vec) * sigma / math.sqrt(D)
    result = mat_or_vec + noise
    return torch.nan_to_num(result, nan=0.0, posinf=clip_norm, neginf=-clip_norm)


def clip_single_gradient(gradient: torch.Tensor, clip_norm: float = DP_CLIP_NORM) -> torch.Tensor:
    """Per-client L2 norm clip — the same clip defense_dp_sgd applies per-row."""
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=clip_norm, neginf=-clip_norm)
    norm = gradient.norm()
    scale = torch.clamp(clip_norm / (norm + 1e-8), max=1.0)
    return gradient * scale


def dp_sgd_client_transform(
    gradient: torch.Tensor,
    noise_multiplier: float = DP_NOISE_MULTIPLIER,
    clip_norm: float = DP_CLIP_NORM,
) -> torch.Tensor:
    """
    Local-DP-style per-client transform: clip THEN add noise to this single
    client's gradient, as if it were noised before ever leaving the client.

    This is what an eavesdropper/curious server would actually observe for
    this client if DP-SGD is the round's chosen defense. Used exclusively
    by the GIA privacy evaluation (gia.py / fl_env.py) — the main
    defense_dp_sgd() aggregation path below is unchanged (it still noises
    the aggregate, which is a separate, valid central-DP design choice for
    the model-update itself).
    """
    clipped = clip_single_gradient(gradient, clip_norm)
    return _dp_clip_and_noise(clipped, noise_multiplier, clip_norm, n_for_noise_scale=1)


def defense_dp_sgd(
    gradients: Dict[int, torch.Tensor],
    noise_multiplier: float = DP_NOISE_MULTIPLIER,
    clip_norm: float = DP_CLIP_NORM,
    device: torch.device = DEVICE,
    **kwargs,
) -> torch.Tensor:
    cids, mat = _stack(gradients)
    N = mat.shape[0]

    mat = torch.nan_to_num(mat, nan=0.0, posinf=clip_norm, neginf=-clip_norm)

    # Per-client L2 norm clipping
    norms   = mat.norm(dim=1, keepdim=True)
    scales  = torch.clamp(clip_norm / (norms + 1e-8), max=1.0)
    clipped = mat * scales

    agg = clipped.mean(dim=0)

    # Gaussian noise on the aggregate — normalised by sqrt(D) to keep noise
    # norm ~ sigma*C regardless of parameter count (central-DP style, applied
    # to the published model update).
    result = _dp_clip_and_noise(agg, noise_multiplier, clip_norm, n_for_noise_scale=N)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# ACTION 1 — FLTrust
# ─────────────────────────────────────────────────────────────────────────────

def compute_server_gradient(
    model: nn.Module,
    root_dataset: Dataset,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Compute gradient of CrossEntropyLoss on the root dataset."""
    model = model.to(device)
    # Work on a copy so we don't modify the global model's .grad
    import copy
    m = copy.deepcopy(model).to(device)
    m.train()
    loader = DataLoader(root_dataset, batch_size=len(root_dataset),
                        shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    m.zero_grad()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    loss = criterion(m(imgs), labels)
    loss.backward()
    grad = torch.cat([
        p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
        for p in m.parameters()
    ])
    return grad.detach()


def defense_fltrust(
    gradients: Dict[int, torch.Tensor],
    global_model: nn.Module,
    root_dataset: Dataset,
    device: torch.device = DEVICE,
    **kwargs,
) -> torch.Tensor:
    """
    FLTrust: weighted aggregation where weights = ReLU(cosine_sim(g_i, g_server)).
    Clients with negative cosine similarity are excluded (weight=0).
    """
    server_grad = compute_server_gradient(global_model, root_dataset, device)
    server_norm = server_grad.norm() + 1e-8

    cids, mat = _stack(gradients)   # (N, D)

    # Cosine similarities
    dot  = mat @ server_grad                                      # (N,)
    norms = mat.norm(dim=1) + 1e-8                               # (N,)
    cos  = dot / (norms * server_norm)                           # (N,)

    weights = torch.relu(cos)                                    # (N,)
    w_sum   = weights.sum()

    if w_sum < 1e-8:
        # All clients have negative similarity — fall back to server gradient
        return server_grad

    weights = weights / w_sum                                    # normalise
    agg = (weights.unsqueeze(1) * mat).sum(dim=0)               # (D,)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# ACTION 2 — Multi-Krum
# ─────────────────────────────────────────────────────────────────────────────

def defense_multi_krum(
    gradients: Dict[int, torch.Tensor],
    f: int = KRUM_F,
    **kwargs,
) -> torch.Tensor:
    """
    Multi-Krum: select n-f clients with smallest Krum score and average them.

    Krum score(i) = sum of distances to f+1 nearest neighbors (excluding self).
    This tolerates up to f Byzantine clients.
    """
    cids, mat = _stack(gradients)   # (N, D)
    N = mat.shape[0]
    m = max(N - f, 1)               # number of clients to keep

    # Pairwise squared L2 distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq_norms = (mat ** 2).sum(dim=1)                              # (N,)
    dist_sq  = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) \
               - 2 * mat @ mat.T                                  # (N, N)
    dist_sq  = torch.clamp(dist_sq, min=0.0)
    dist     = dist_sq.sqrt()                                     # (N, N)

    # Set diagonal to infinity so a client doesn't count itself
    inf_mask = torch.full_like(dist, float('inf'))
    dist     = torch.where(torch.eye(N, device=dist.device).bool(), inf_mask, dist)

    # For each client: sum of distances to f+1 nearest neighbours
    neighbors = f + 1
    sorted_d, _ = dist.sort(dim=1)
    krum_scores = sorted_d[:, :neighbors].sum(dim=1)             # (N,)

    # Select m clients with smallest score
    _, selected = krum_scores.topk(m, largest=False)
    kept = mat[selected]                                          # (m, D)
    return kept.mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION 3 — FLAME
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE (fix, see project changelog): the original implementation used
# hdbscan.HDBSCAN on the raw (or precomputed-cosine-distance) gradient
# matrix. Empirically, HDBSCAN cannot find any cluster at N=10 points
# regardless of min_cluster_size/min_samples/metric — every point gets
# labelled noise (-1), so the code fell back to defense_multi_krum() on
# EVERY round, meaning "FLAME" was silently identical to Multi-Krum in
# all prior results. Fixed by using scipy agglomerative clustering
# (average linkage) on the cosine-distance matrix instead, cutting the
# dendrogram into 2 groups and keeping the larger one. This is a
# density-free method and works reliably at small N.

def defense_flame(
    gradients: Dict[int, torch.Tensor],
    min_cluster_size: int = FLAME_MIN_CLUSTER,
    device: torch.device = DEVICE,
    **kwargs,
) -> torch.Tensor:
    """
    FLAME: cluster client gradients (cosine-distance, agglomerative), keep
    the majority cluster, apply adaptive noise proportional to the cosine
    spread within the kept cluster.

    Falls back to Multi-Krum only for genuine edge cases (too few clients
    to form two groups).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    cids, mat = _stack(gradients)          # (N, D)
    N = mat.shape[0]

    mat_np = mat.cpu().float().numpy()
    mat_np = np.nan_to_num(mat_np, nan=0.0, posinf=1e6, neginf=-1e6)
    mat_np = np.clip(mat_np, -1e6, 1e6)

    norms = np.linalg.norm(mat_np, axis=1, keepdims=True) + 1e-8
    row_norms = norms.squeeze(1)
    valid_mask = row_norms > 1e-8

    if N < 4 or valid_mask.sum() < 4:
        # Too few usable points to meaningfully split into 2 groups
        warnings.warn(
            f"defense_flame: only {int(valid_mask.sum())}/{N} usable gradients "
            f"this round — falling back to Multi-Krum.",
            RuntimeWarning,
        )
        return defense_multi_krum(gradients, **kwargs)

    mat_cos = mat_np / norms

    # Pairwise cosine distance matrix (bounded [0, 2], 0 = identical direction)
    cos_sim  = mat_cos @ mat_cos.T
    cos_dist = np.clip(1.0 - cos_sim, 0.0, None)
    np.fill_diagonal(cos_dist, 0.0)
    # Symmetrize defensively against float asymmetry before squareform
    cos_dist = (cos_dist + cos_dist.T) / 2.0

    condensed = squareform(cos_dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=2, criterion='maxclust')  # 1 or 2

    unique, counts = np.unique(labels, return_counts=True)
    majority_label = unique[counts.argmax()]
    mask = labels == majority_label
    kept_np = mat_np[mask]
    kept = torch.tensor(kept_np, device=device, dtype=mat.dtype)

    # Adaptive noise: calibrated to cosine spread within kept cluster
    kept_norm = kept / (kept.norm(dim=1, keepdim=True) + 1e-8)
    cos_mat = kept_norm @ kept_norm.T
    k = kept.shape[0]
    if k > 1:
        cos_sim_mean = (cos_mat.sum() - k) / (k * (k - 1))
    else:
        cos_sim_mean = torch.tensor(1.0, device=device)

    noise_scale = float(torch.clamp(1.0 - cos_sim_mean, min=0.0, max=1.0))
    agg = kept.mean(dim=0)
    if noise_scale > 1e-4:
        noise = torch.randn_like(agg) * noise_scale * DP_CLIP_NORM
        agg = agg + noise

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH TABLE
# ─────────────────────────────────────────────────────────────────────────────

DEFENSE_NAMES = {
    0: "DP-SGD",
    1: "FLTrust",
    2: "Multi-Krum",
    3: "FLAME",
    4: "NoDefense",
}


def apply_defense(
    action: int,
    gradients: Dict[int, torch.Tensor],
    global_model: nn.Module,
    root_dataset: Optional[Dataset] = None,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Dispatch to the correct defense given `action` (0-4).

    Never apply two defenses simultaneously (spec §9.6).
    """
    if action == 0:
        return defense_dp_sgd(gradients, device=device)
    elif action == 1:
        if root_dataset is None:
            raise ValueError("FLTrust requires root_dataset")
        return defense_fltrust(gradients, global_model=global_model,
                               root_dataset=root_dataset, device=device)
    elif action == 2:
        return defense_multi_krum(gradients)
    elif action == 3:
        return defense_flame(gradients, device=device)
    elif action == 4:
        return defense_no_defense(gradients)
    else:
        raise ValueError(f"Unknown action: {action}. Must be in [0,4].")


def apply_client_level_defense_transform(
    action: int,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """
    Return the gradient AS TRANSMITTED by a single client, under the given
    defense action — i.e. only the part of the defense that happens at/before
    the client, before server-side aggregation/filtering.

    This exists specifically for GIA privacy evaluation (gia.py), which
    attacks one client's individual gradient. Server-side-only defenses
    (FLTrust reweighting, Multi-Krum selection, FLAME clustering) do not
    change what a client transmits, so they are identity here — matching
    the spec's own "best against" table, where only DP-SGD is claimed to
    protect against gradient inversion. Only DP-SGD perturbs the gradient
    itself, so only DP-SGD is non-identity here.
    """
    if action == 0:
        return dp_sgd_client_transform(gradient)
    # actions 1 (FLTrust), 2 (Multi-Krum), 3 (FLAME), 4 (NoDefense):
    # no per-client perturbation happens before transmission.
    return gradient


# ─────────────────────────────────────────────────────────────────────────────
# APPLY AGGREGATED GRADIENT TO GLOBAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def apply_gradient_to_model(
    model: nn.Module,
    gradient: torch.Tensor,
    lr: float = 1.0,
) -> None:
    grad_norm = gradient.norm()
    max_norm  = 50.0
    if grad_norm > max_norm:
        gradient = gradient * (max_norm / (grad_norm + 1e-8))

    weight_decay = 1e-4

    offset = 0
    for p in model.parameters():
        numel = p.numel()
        delta = gradient[offset: offset + numel].view_as(p.data)
        # Apply gradient update with weight decay
        p.data.mul_(1.0 - weight_decay)
        p.data.sub_(lr * delta)
        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)
        offset += numel
# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    D = 1000
    N = 10
    torch.manual_seed(0)
    fake_grads = {i: torch.randn(D) for i in range(N)}
    # Inject 2 Byzantine-scale gradients
    fake_grads[0] = torch.randn(D) * 50
    fake_grads[1] = torch.randn(D) * 50

    for act in range(NUM_DEFENSES):
        if act == 1:
            continue  # skip FLTrust (needs model + root dataset)
        result = apply_defense(act, fake_grads, global_model=None)
        print(f"Action {act} ({DEFENSE_NAMES[act]}): norm={result.norm():.4f}")

    print("\n✓ defenses self-test passed (FLTrust skipped — needs model).")