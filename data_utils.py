"""
data_utils.py — Dataset loading, Dirichlet Non-IID partitioning, backdoor
                trigger injection, and FLTrust root-dataset sampling.

Key guarantees
──────────────
• Every client receives ≥ MIN_SAMPLES_PER_CLIENT samples (oversampled if needed).
• Partitioning is deterministic given (alpha, seed).
• Backdoor trigger is a 3×3 white patch at bottom-right, applied at DATA level
  (not inside the model), relabeled to BACKDOOR_TARGET_CLASS.
• The FLTrust root dataset is sampled once, IID, and never modified.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional

from config import (
    DATA_DIR, NUM_CLIENTS, MIN_SAMPLES_PER_CLIENT, LOCAL_BATCH_SIZE,
    BACKDOOR_TRIGGER_SIZE, BACKDOOR_POISON_RATE, BACKDOOR_TARGET_CLASS,
    FLTRUST_ROOT_SIZE, NUM_CLASSES, DEVICE
)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# BACKDOOR DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BackdoorDataset(Dataset):
    """
    Wraps an existing dataset and poisons a fixed fraction of samples by
    stamping a 3×3 white patch at the bottom-right corner of the TENSOR
    (after ToTensor + Normalize) and relabeling to BACKDOOR_TARGET_CLASS.

    The trigger is applied in normalised pixel space:
        white = (1 - mean) / std  per channel
    so that it appears as maximum brightness after normalisation.
    """

    TRIGGER_VALUE = torch.tensor(
        [
            (1.0 - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
            for c in range(3)
        ]
    ).view(3, 1, 1)  # broadcast over H×W patch

    def __init__(self, base_dataset: Dataset, poison_rate: float = BACKDOOR_POISON_RATE,
                 target_class: int = BACKDOOR_TARGET_CLASS,
                 trigger_size: int = BACKDOOR_TRIGGER_SIZE,
                 seed: int = 0):
        self.base    = base_dataset
        self.rate    = poison_rate
        self.target  = target_class
        self.tsz     = trigger_size

        rng = np.random.RandomState(seed)
        n   = len(base_dataset)
        self.poisoned_indices = set(
            rng.choice(n, size=int(n * poison_rate), replace=False).tolist()
        )

    def _stamp_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) float tensor.  Stamp white patch in-place (on a copy)."""
        img = img.clone()
        h, w = img.shape[1], img.shape[2]
        img[:, h - self.tsz:h, w - self.tsz:w] = self.TRIGGER_VALUE.expand(
            3, self.tsz, self.tsz
        )
        return img

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poisoned_indices:
            img   = self._stamp_trigger(img)
            label = self.target
        return img, label


class TriggerOnlyDataset(Dataset):
    """
    Test-time dataset: stamps the trigger on EVERY sample and sets label to
    target_class.  Used to compute ASR (Attack Success Rate).
    """
    def __init__(self, base_dataset: Dataset,
                 trigger_size: int = BACKDOOR_TRIGGER_SIZE,
                 target_class: int = BACKDOOR_TARGET_CLASS):
        self.base    = base_dataset
        self.tsz     = trigger_size
        self.target  = target_class
        self.trigger_val = BackdoorDataset.TRIGGER_VALUE

    def _stamp(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clone()
        h, w = img.shape[1], img.shape[2]
        img[:, h - self.tsz:h, w - self.tsz:w] = self.trigger_val.expand(
            3, self.tsz, self.tsz
        )
        return img

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self._stamp(img), self.target


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10() -> Tuple[Dataset, Dataset]:
    """Download (if needed) and return (train_dataset, test_dataset)."""
    train = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test  = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# FAST LABEL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE (fix, see project changelog): the previous implementation built the
# label array via `[dataset[i][1] for i in range(len(dataset))]`, which
# runs the FULL transform pipeline (RandomCrop, RandomHorizontalFlip,
# ToTensor, Normalize) on all ~50,000 images just to read an int label.
# dirichlet_partition() is called on every episode reset, so this cost was
# being paid every episode, not once. torchvision's CIFAR10 already
# exposes labels directly via `.targets` — use that when available.

def _get_labels(dataset) -> np.ndarray:
    """Return integer labels for `dataset` without running its transform."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    # Fallback for wrapped/custom datasets with no direct label attribute —
    # slow path, but only reached if the above don't exist.
    return np.array([dataset[i][1] for i in range(len(dataset))])


# ─────────────────────────────────────────────────────────────────────────────
# DIRICHLET PARTITIONING
# ─────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(
    dataset: Dataset,
    num_clients: int = NUM_CLIENTS,
    alpha: float = 0.1,
    min_samples: int = MIN_SAMPLES_PER_CLIENT,
    seed: int = 42,
    num_classes: int = NUM_CLASSES,
) -> List[List[int]]:
    """
    Partition dataset indices among `num_clients` using a Dirichlet(alpha)
    distribution over class labels.

    Guarantees:
    • Every client has ≥ min_samples indices.
    • Shortfalls are fixed by oversampling from the client's existing pool
      (stratified: preserves class distribution within the client).
    • Never drops a client.

    Returns:
        List of index-lists, one per client.
    """
    rng = np.random.RandomState(seed)

    # Collect indices per class
    labels = _get_labels(dataset)
    class_indices: Dict[int, List[int]] = {
        c: np.where(labels == c)[0].tolist() for c in range(num_classes)
    }
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Draw Dirichlet proportions: shape (num_classes, num_clients)
    proportions = rng.dirichlet(alpha=[alpha] * num_clients, size=num_classes)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs  = class_indices[c]
        n     = len(idxs)
        props = proportions[c]                    # sum-to-1 over clients
        # Convert to counts
        counts = (props * n).astype(int)
        # Fix rounding so total == n
        diff = n - counts.sum()
        # Add remainder to top-diff clients by proportion
        order = np.argsort(-props)
        for i in range(abs(diff)):
            counts[order[i % num_clients]] += int(np.sign(diff))

        start = 0
        for client_id, cnt in enumerate(counts):
            client_indices[client_id].extend(idxs[start: start + cnt])
            start += cnt

    # Enforce minimum samples via oversampling
    for client_id in range(num_clients):
        pool = client_indices[client_id]
        if len(pool) < min_samples:
            shortage = min_samples - len(pool)
            # Oversample with replacement from existing pool
            extra = rng.choice(pool, size=shortage, replace=True).tolist()
            client_indices[client_id] = pool + extra

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_client_datasets(
    train_dataset: Dataset,
    client_indices: List[List[int]],
    backdoor_client_ids: List[int],
    seed: int = 42,
) -> List[Dataset]:
    """
    Given per-client index lists, wrap each client's data in a Subset.
    For backdoor clients, further wrap with BackdoorDataset.
    """
    datasets = []
    for cid, idxs in enumerate(client_indices):
        subset = _IndexedSubset(train_dataset, idxs)
        if cid in backdoor_client_ids:
            subset = BackdoorDataset(subset, seed=seed + cid)
        datasets.append(subset)
    return datasets


class _IndexedSubset(Dataset):
    """Like torch Subset but supports arbitrary (possibly repeated) indices."""
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def get_client_loader(client_dataset: Dataset,
                      batch_size: int = LOCAL_BATCH_SIZE,
                      shuffle: bool = True) -> DataLoader:
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def get_test_loader(test_dataset: Dataset, batch_size: int = 256) -> DataLoader:
    # FIX (see project changelog): this loader is re-iterated fresh every
    # single round (evaluate_accuracy/evaluate_asr/evaluate_loss all run
    # every round, and PyTorch DataLoader with num_workers>0 spawns new
    # worker processes on each fresh iteration, not just once). Over a
    # 200-round episode that's ~400-600 process spawns for eval alone —
    # the same category of issue client.py's docstring already identifies
    # as "Windows DataLoader handle exhaustion... during long runs", just
    # not fixed here. num_workers=0 avoids it; DEVICE is imported above.
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST ROOT DATASET
# ─────────────────────────────────────────────────────────────────────────────

def sample_fltrust_root(
    train_dataset: Dataset,
    size: int = FLTRUST_ROOT_SIZE,
    seed: int = 42,
) -> Dataset:
    """
    Sample `size` indices IID (balanced over classes) from train_dataset.
    This is sampled once and must NOT be modified across rounds.
    """
    rng   = np.random.RandomState(seed)
    labels = _get_labels(train_dataset)
    per_class = size // NUM_CLASSES
    chosen = []
    for c in range(NUM_CLASSES):
        class_idxs = np.where(labels == c)[0]
        chosen.extend(rng.choice(class_idxs, size=per_class, replace=False).tolist())
    # Fill up to `size` if not divisible
    remaining = size - len(chosen)
    if remaining > 0:
        pool = list(set(range(len(train_dataset))) - set(chosen))
        chosen.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    return _IndexedSubset(train_dataset, chosen)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL DISTRIBUTION UTILITY (for state computation)
# ─────────────────────────────────────────────────────────────────────────────

def client_label_distribution(client_dataset: Dataset,
                               num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Return normalised label histogram for a client's dataset.

    FIX (see project changelog): the original `for _, label in
    client_dataset` iterates __getitem__ for every sample, which — same
    issue as dirichlet_partition/sample_fltrust_root — runs the full
    train_transform pipeline just to read a label, and this function is
    called once per client (20x) on every episode reset. We special-case
    the two dataset wrapper types actually produced by
    build_client_datasets() to read labels directly and vectorized;
    anything else falls back to the original (correct, just slower) path.
    """
    if isinstance(client_dataset, _IndexedSubset):
        labels = _get_labels(client_dataset.dataset)[client_dataset.indices]
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    elif isinstance(client_dataset, BackdoorDataset) and isinstance(client_dataset.base, _IndexedSubset):
        base = client_dataset.base
        labels = _get_labels(base.dataset)[base.indices].copy()
        poisoned = np.array(sorted(client_dataset.poisoned_indices), dtype=int)
        if poisoned.size:
            labels[poisoned] = client_dataset.target
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    else:
        counts = np.zeros(num_classes, dtype=np.float64)
        for _, label in client_dataset:
            counts[label] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def estimate_alpha_mle(
    label_dists: List[np.ndarray],
    num_iterations: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Maximum-likelihood estimate of Dirichlet concentration α from observed
    per-client label distributions, using the fixed-point iteration of
    Minka (2000).  Returns a scalar α (assumes symmetric Dirichlet).
    """
    K = label_dists[0].shape[0]
    N = len(label_dists)
    # Clamp to avoid log(0)
    dists = np.clip(np.array(label_dists), 1e-8, 1.0)
    dists = dists / dists.sum(axis=1, keepdims=True)

    # Start with method-of-moments estimate
    mean = dists.mean(axis=0)
    var  = dists.var(axis=0).mean()
    mean_mean = mean.mean()
    alpha = max(mean_mean * (mean_mean * (1 - mean_mean) / max(var, 1e-8) - 1), 0.01)

    for _ in range(num_iterations):
        log_p_bar = np.log(dists).mean(axis=0).mean()  # mean over clients & classes
        # Symmetric Digamma fixed-point update (Minka 2000).
        # FIX: denominator must be scaled by N*K to match the numerator's
        # N*K scaling (previously only scaled by N, missing a factor of K).
        # Confirmed empirically: the old formula saturated at the alpha=10.0
        # clip ceiling for every true alpha in {0.1, 0.3, 0.5, 1.0} tested,
        # making the alpha_estimate state feature a constant (zero
        # information). The corrected version recovers ~0.12/0.29/0.46/1.10
        # for true alpha 0.1/0.3/0.5/1.0 respectively.
        from scipy.special import digamma, polygamma
        num   = N * K * (digamma(K * alpha) - digamma(alpha))
        denom = -N * K * log_p_bar + 1e-10
        alpha_new = alpha * num / denom
        alpha_new = max(alpha_new, 1e-3)
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return float(np.clip(alpha, 0.01, 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading CIFAR-10 …")
    tr, te = load_cifar10()
    print(f"  Train: {len(tr)}   Test: {len(te)}")

    print("Partitioning with α=0.1 …")
    parts = dirichlet_partition(tr, num_clients=20, alpha=0.1, seed=42)
    sizes = [len(p) for p in parts]
    print(f"  Client sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}")
    assert min(sizes) >= MIN_SAMPLES_PER_CLIENT, "Min-sample guarantee violated!"

    print("Building client datasets …")
    c_datasets = build_client_datasets(tr, parts, backdoor_client_ids=[0, 1])
    print(f"  Client 0 (backdoor) size: {len(c_datasets[0])}")

    print("Sampling FLTrust root …")
    root = sample_fltrust_root(tr)
    print(f"  Root size: {len(root)}")

    print("Estimating α via MLE …")
    label_dists = [client_label_distribution(d) for d in c_datasets]
    alpha_hat = estimate_alpha_mle(label_dists)
    print(f"  True α=0.1  →  MLE α̂={alpha_hat:.4f}")

    print("\n✓ data_utils self-test passed.")