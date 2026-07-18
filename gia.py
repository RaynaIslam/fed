"""
gia.py — Gradient Inversion Attack simulation using the DLG method.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from config import (
    GIA_EVAL_INTERVAL, GIA_OPT_STEPS, GIA_LR, DEVICE, NUM_CLASSES, GIA_TV_WEIGH
)
from models import clone_model
from defenses import apply_client_level_defense_transform
from data_utils import CIFAR10_MEAN, CIFAR10_STD

# Valid pixel range in NORMALISED space: [0,1] raw pixel maps to
# [(0-mean)/std, (1-mean)/std] per channel. Clamping the dummy image to this
# box after every optimiser step keeps the search inside physically-possible
# images instead of wandering into normalised values no real photo can reach
# — a standard stabiliser for gradient-inversion attacks.
_CHANNEL_LOW = torch.tensor(
    [(0.0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)]
).view(1, 3, 1, 1)
_CHANNEL_HIGH = torch.tensor(
    [(1.0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)]
).view(1, 3, 1, 1)


def _total_variation(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV norm — mean absolute difference between adjacent pixels."""
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return dh + dw


def _compute_gradient_with_grad(
    model: nn.Module,
    img: torch.Tensor,
    label: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    output = model(img)
    loss   = criterion(output, label.to(device))
    grads  = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=True,
        allow_unused=True,
    )
    flat = torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel(), device=device)
        for g, p in zip(grads, model.parameters())
    ])
    return flat


def _compute_gradient_no_graph(
    model: nn.Module,
    img: torch.Tensor,
    label: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    out  = model(img.detach())
    loss = criterion(out, label.to(device))
    loss.backward()
    grad = torch.cat([
        p.grad.view(-1) if p.grad is not None
        else torch.zeros(p.numel(), device=device)
        for p in model.parameters()
    ])
    return grad.detach()


def dlg_reconstruct(
    model: nn.Module,
    true_gradient: torch.Tensor,
    true_img: torch.Tensor,
    true_label: torch.Tensor,
    num_steps: int = GIA_OPT_STEPS,
    lr: float = GIA_LR,
    tv_weight: float = GIA_TV_WEIGHT,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, float]:
    true_gradient = true_gradient.to(device).detach()
    ch_low  = _CHANNEL_LOW.to(device)
    ch_high = _CHANNEL_HIGH.to(device)

    dummy_img = torch.randn_like(true_img.unsqueeze(0)).to(device)
    dummy_img.requires_grad_(True)

    # FIX (see project changelog): label recovery via the iDLG closed-form
    # trick (Zhao et al., "iDLG: Improved Deep Leakage from Gradients",
    # 2020) instead of the previous try-all-classes loop. For a single-
    # sample cross-entropy loss, the gradient w.r.t. the TRUE class's final-
    # layer bias is the only negative entry among the num_classes bias
    # gradients; every other class's entry is non-negative. This makes
    # label recovery a single argmin over a length-10 slice — no cloning,
    # no forward/backward passes at all — replacing what was previously
    # NUM_CLASSES (10) full model clones + forward + backward per GIA
    # evaluation. Confirmed the final-layer bias occupies exactly the last
    # NUM_CLASSES elements of the flat parameter/gradient vector for this
    # architecture (see project changelog for the exact offsets).
    final_bias_grad = true_gradient[-NUM_CLASSES:]
    dummy_label_val = int(torch.argmin(final_bias_grad).item())
    dummy_label = torch.tensor([dummy_label_val], device=device, dtype=torch.long)

    # Main reconstruction model — params need grad so graph flows to dummy_img
    m = clone_model(model).to(device)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [dummy_img],
        lr=lr,
        max_iter=20,
        history_size=50,
        tolerance_grad=1e-7,
        line_search_fn='strong_wolfe',
    )

    # FIX (see project changelog "open item" — GIA attack strength): plain
    # gradient-matching loss alone tends to converge to noisy, unstructured
    # images against pooled CNNs (max-pooling destroys a lot of the gradient
    # signal DLG relies on). Total-variation regularisation (Geiping et al.,
    # "Inverting Gradients", 2020) adds a natural-image prior that penalises
    # high-frequency noise in the reconstruction, which is the standard fix
    # for exactly this failure mode. Combined with clamping to the valid
    # pixel range after each step (below), this keeps the optimiser inside
    # physically-plausible images instead of an unconstrained search.
    def closure():
        optimizer.zero_grad()
        dummy_grad = _compute_gradient_with_grad(m, dummy_img, dummy_label, device)
        grad_loss  = ((dummy_grad - true_gradient) ** 2).sum()
        tv_loss    = _total_variation(dummy_img)
        rec_loss   = grad_loss + tv_weight * tv_loss
        rec_loss.backward(inputs=[dummy_img])
        return rec_loss

    best_mse = float('inf')
    best_img = dummy_img.detach().clone()

    lbfgs_steps = max(1, num_steps // 20)
    for _ in range(lbfgs_steps):
        optimizer.step(closure)
        with torch.no_grad():
            dummy_img.clamp_(ch_low, ch_high)
            mse = (
                (dummy_img.detach() - true_img.unsqueeze(0).to(device)) ** 2
            ).mean()
            if mse.item() < best_mse:
                best_mse = mse.item()
                best_img = dummy_img.detach().clone()

    return best_img.squeeze(0).cpu(), best_mse


class GIAManager:
    def __init__(
        self,
        eval_interval: int = GIA_EVAL_INTERVAL,
        num_steps: int = GIA_OPT_STEPS,
        device: torch.device = DEVICE,
    ):
        self.eval_interval = eval_interval
        self.num_steps     = num_steps
        self.device        = device
        self.history: list = []
        self._baseline_mse: Optional[float] = None

    def should_evaluate(self, round_idx: int) -> bool:
        return (round_idx % self.eval_interval) == 0

    def evaluate(
        self,
        round_idx: int,
        model: nn.Module,
        gia_context: Dict[int, tuple],
        action: int,
    ) -> float:
        """
        `action`: the defense chosen THIS round. The gradient we attack is
        the one the client would actually have transmitted under that
        defense — see defenses.apply_client_level_defense_transform().
        Previously this always attacked the raw, pre-defense gradient
        regardless of `action`, which decoupled privacy_score from the
        agent's choice entirely. Only DP-SGD perturbs an individual
        client's gradient before transmission (matches the spec's "best
        against" table), so this is the only action expected to move the
        privacy score meaningfully.
        """
        if not gia_context:
            return self._last_raw_mse()

        mses = []
        for cid, ctx in gia_context.items():
            gradient, imgs, labels = ctx
            transmitted_gradient = apply_client_level_defense_transform(action, gradient)
            true_img   = imgs[0]
            true_label = labels[0:1]
            _, mse = dlg_reconstruct(
                model=model,
                true_gradient=transmitted_gradient,
                true_img=true_img,
                true_label=true_label,
                num_steps=self.num_steps,
                device=self.device,
            )
            mses.append(mse)

        raw_mse = float(np.mean(mses)) if mses else self._last_raw_mse()
        self.history.append((round_idx, raw_mse))

        if self._baseline_mse is None:
            self._baseline_mse = max(raw_mse, 1e-8)

        return raw_mse

    def get_normalised_mse(self, round_idx: int) -> float:
        raw = self._get_interpolated_mse(round_idx)
        if self._baseline_mse is None or self._baseline_mse < 1e-10:
            return 0.5
        return float(np.clip(raw / self._baseline_mse, 0.0, 1.0))

    def get_privacy_score(self, round_idx: int) -> float:
        return 1.0 - self.get_normalised_mse(round_idx)

    def _last_raw_mse(self) -> float:
        if self.history:
            return self.history[-1][1]
        return 1.0

    def _get_interpolated_mse(self, round_idx: int) -> float:
        if not self.history:
            return 1.0
        if len(self.history) == 1:
            return self.history[0][1]
        rounds = [h[0] for h in self.history]
        mses   = [h[1] for h in self.history]
        if round_idx <= rounds[0]:
            return mses[0]
        if round_idx >= rounds[-1]:
            return mses[-1]
        for i in range(len(rounds) - 1):
            if rounds[i] <= round_idx <= rounds[i + 1]:
                t = (round_idx - rounds[i]) / max(rounds[i + 1] - rounds[i], 1)
                return mses[i] + t * (mses[i + 1] - mses[i])
        return mses[-1]


if __name__ == "__main__":
    from models import get_model

    print("Testing DLG reconstruction (small model, single image) ...")
    model = get_model()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    torch.manual_seed(0)
    img   = torch.randn(3, 32, 32)
    label = torch.tensor([3], dtype=torch.long)

    m_tmp = clone_model(model).to(DEVICE)
    m_tmp.train()
    for p in m_tmp.parameters():
        p.requires_grad_(True)
    true_grad = _compute_gradient_no_graph(m_tmp, img.unsqueeze(0), label, DEVICE)

    recon, mse = dlg_reconstruct(
        model, true_grad, img, label,
        num_steps=40, device=DEVICE,
    )
    print(f"  Reconstruction MSE: {mse:.6f}")
    print(f"  Recon shape: {recon.shape}")

    mgr = GIAManager(eval_interval=5)
    gia_ctx = {0: (true_grad, img.unsqueeze(0), label)}
    raw = mgr.evaluate(0, model, gia_ctx, action=4)
    print(f"  GIA manager raw MSE at round 0: {raw:.6f}")
    print(f"  Privacy score: {mgr.get_privacy_score(0):.4f}")
    print("\n✓ gia self-test passed.")
