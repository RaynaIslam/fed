"""
client.py — Local training logic for each federated client.

Responsibilities
────────────────
• Receive a fresh copy of the global model each round (enforced by caller).
• Train for LOCAL_EPOCHS epochs on local data with SGD.
• Return the gradient (delta = trained_params − initial_params).
• Byzantine clients return a scaled/flipped gradient instead.
• GIA-passive clients record their gradients for reconstruction attacks.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    LOCAL_EPOCHS, LOCAL_LR, LOCAL_MOMENTUM, LOCAL_BATCH_SIZE,
    BYZANTINE_SCALE, DEVICE, GIA_BATCH_SIZE
)
from models import get_gradients, clone_model


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def local_train(
    global_model: nn.Module,
    client_dataset: Dataset,
    client_id: int,
    is_byzantine: bool = False,
    device: torch.device = DEVICE,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LOCAL_LR,
    momentum: float = LOCAL_MOMENTUM,
    batch_size: int = LOCAL_BATCH_SIZE,
) -> Tuple[torch.Tensor, float]:
    model = clone_model(global_model).to(device)
    model.train()

    init_params = torch.cat(
        [p.data.view(-1) for p in model.parameters()]
    ).clone()

    # ── Safe DataLoader for long-running Windows processes ──────────────
    actual_batch = min(batch_size, max(1, len(client_dataset)))
    
    # Pre-load all data into memory to avoid DataLoader handle exhaustion
    # on Windows during long training runs
    all_imgs, all_labels = [], []
    tmp_loader = DataLoader(
        client_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    for imgs, labels in tmp_loader:
        all_imgs.append(imgs)
        all_labels.append(labels)
    
    if not all_imgs:
        # Empty dataset — return zero gradient
        zero_grad = torch.zeros_like(init_params)
        return zero_grad, 0.0
    
    all_imgs   = torch.cat(all_imgs,   dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    n_samples  = len(all_imgs)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum
    )

    total_loss  = 0.0
    total_steps = 0

    for epoch in range(local_epochs):
        # Manual shuffle each epoch
        perm = torch.randperm(n_samples)
        all_imgs_e   = all_imgs[perm]
        all_labels_e = all_labels[perm]
        
        for start in range(0, n_samples, actual_batch):
            end    = min(start + actual_batch, n_samples)
            imgs   = all_imgs_e[start:end].to(device)
            labels = all_labels_e[start:end].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss  += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)

    trained_params = torch.cat(
        [p.data.view(-1) for p in model.parameters()]
    )
    gradient = init_params - trained_params

    # FIX (see project changelog): if this client's local training diverged,
    # `gradient` can contain NaN/inf. The old code only guarded against this
    # via `grad_norm = gradient.norm(); if grad_norm > max_norm: clip` — but
    # NaN comparisons are ALWAYS False in Python/PyTorch (`nan > 50.0` is
    # `False`), so a NaN gradient silently skips the clip and gets returned
    # completely unsanitized. Once even one client returns a NaN gradient,
    # unsanitized aggregation paths (defense_no_defense's plain mean, among
    # others) propagate it into the aggregated update; fl_env.py's own
    # nan_to_num on the aggregate then zeros it out, but repeated rounds of
    # this can leave weights clamped at extreme-but-technically-finite
    # values that overflow in the NEXT forward pass — reproducing NaN again,
    # every round, with no recovery. Confirmed in a real training run: once
    # this state was hit, accuracy froze at a constant ~0.101 (a degenerate,
    # single-class predictor) for 60+ consecutive rounds, and FLAME's
    # "N usable gradients" warning (see defenses.py) reported 0-1/10 usable
    # gradients — consistent with nearly every client training from the same
    # already-corrupted global model simultaneously.
    #
    # Explicit isnan/isinf check first: if this client's gradient is bad,
    # abstain (return a zero gradient) rather than let it poison the round.
    if torch.isnan(gradient).any() or torch.isinf(gradient).any():
        zero_grad = torch.zeros_like(gradient)
        return zero_grad, float('nan')  # keep loss=nan for visibility in logs

    # Clip by L2 norm (not element-wise) — standard FL practice
    grad_norm = gradient.norm()
    max_norm  = 50.0
    if grad_norm > max_norm:
        gradient = gradient * (max_norm / (grad_norm + 1e-8))

    if is_byzantine:
        gradient = gradient * BYZANTINE_SCALE
        byz_norm = gradient.norm()
        byz_max  = 500.0
        if byz_norm > byz_max:
            gradient = gradient * (byz_max / (byz_norm + 1e-8))

    return gradient.detach(), avg_loss

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STEP GRADIENT  (for GIA / DLG reconstruction attacks ONLY)
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE (fix, see project changelog): local_train() returns the CUMULATIVE
# delta over LOCAL_EPOCHS epochs (~35 SGD steps with momentum) of the
# client's full local dataset. Standard DLG-style reconstruction (gia.py)
# assumes it is inverting a SINGLE forward/backward pass on a SINGLE small
# batch — that assumption does not hold for the multi-step delta, so the
# attack was never able to converge regardless of which defense was chosen.
# This function produces the single-step gradient DLG actually needs. The
# multi-epoch delta from local_train() is still what's used for real FL
# aggregation/utility; this is only used for the privacy side-channel.

def compute_single_step_gradient(
    global_model: nn.Module,
    client_dataset: Dataset,
    batch_size: int = GIA_BATCH_SIZE,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One forward/backward pass (no optimizer step) on one small batch.

    Returns (flat_gradient, imgs_batch, labels_batch) — the batch is
    returned too since dlg_reconstruct() needs the true image/label to
    score reconstruction quality against.
    """
    model = clone_model(global_model).to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    actual_batch = min(batch_size, max(1, len(client_dataset)))
    loader = DataLoader(client_dataset, batch_size=actual_batch, shuffle=True, num_workers=0)
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)

    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    loss = criterion(model(imgs), labels)
    loss.backward()

    grad = torch.cat([
        p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
        for p in model.parameters()
    ])
    return grad.detach(), imgs.detach().cpu(), labels.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH CLIENT TRAINING  (called once per FL round by the server)
# ─────────────────────────────────────────────────────────────────────────────

def collect_gradients(
    global_model: nn.Module,
    selected_client_ids: list[int],
    client_datasets: list[Dataset],
    byzantine_client_ids: list[int],
    gia_client_ids: list[int],
    device: torch.device = DEVICE,
) -> Tuple[
    dict[int, torch.Tensor],   # cid → gradient
    dict[int, float],          # cid → train loss
    dict[int, tuple],          # cid → (gradient, sample_batch) for GIA
]:
    """
    Run local training for all selected clients and return their gradients.

    Also returns GIA context: for clients in `gia_client_ids` that are
    currently selected, we store (gradient, one_batch) so the GIA module
    can attempt reconstruction.

    Parameters
    ──────────
    global_model       : current global model (read-only; deep-copied internally)
    selected_client_ids: which clients participate this round
    client_datasets    : list indexed by client_id
    byzantine_client_ids: clients that submit adversarial gradients
    gia_client_ids     : clients whose gradients are recorded for GIA
    """
    gradients: dict[int, torch.Tensor] = {}
    train_losses: dict[int, float]     = {}
    gia_context: dict[int, tuple]      = {}

    for cid in selected_client_ids:
        is_byz = cid in byzantine_client_ids
        grad, loss = local_train(
            global_model=global_model,
            client_dataset=client_datasets[cid],
            client_id=cid,
            is_byzantine=is_byz,
            device=device,
        )
        gradients[cid]    = grad
        train_losses[cid] = loss

        # Record GIA context: a SINGLE-STEP gradient + the batch it came
        # from, NOT the multi-epoch local_train() delta above — see
        # compute_single_step_gradient() docstring for why. This is what
        # gia.py's dlg_reconstruct() is actually able to invert.
        if cid in gia_client_ids and not is_byz:
            try:
                single_step_grad, sample_imgs, sample_labels = compute_single_step_gradient(
                    global_model=global_model,
                    client_dataset=client_datasets[cid],
                    device=device,
                )
                gia_context[cid] = (single_step_grad, sample_imgs, sample_labels)
            except StopIteration:
                pass

    return gradients, train_losses, gia_context


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLIENT VALIDATION LOSS  (used for state: client_divergence)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_client_val_loss(
    model: nn.Module,
    client_datasets: list[Dataset],
    selected_ids: list[int],
    device: torch.device = DEVICE,
) -> dict[int, float]:
    """Compute validation loss of the current global model on each client's data."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses    = {}
    for cid in selected_ids:
        loader = DataLoader(client_datasets[cid], batch_size=128,
                            shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
        total, steps = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            total  += loss.item()
            steps  += 1
        losses[cid] = total / max(steps, 1)
    return losses