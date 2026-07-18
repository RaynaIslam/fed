"""
fl_env.py — Gymnasium environment wrapping the full FL simulation.

Spec §6.2:
    observation_space : Box(0, 1, shape=(15,), dtype=float32)
    action_space      : Discrete(5)
    reset()  : reinitialise FL (fresh model + fresh partitions)
    step(a)  : run one FL round with defense `a`
               returns (obs, reward, terminated, truncated, info)
    terminated = False always
    truncated  = True after `total_rounds` rounds

Thread-safety note
──────────────────
Each env instance owns its own model, datasets, and RNG state.
Multiple instances can run in parallel (SubprocVecEnv) safely.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from config import (
    NUM_CLIENTS, CLIENTS_PER_ROUND, FULL_ROUNDS, SMOKE_ROUNDS,
    GIA_CLIENTS, BACKDOOR_CLIENTS, BYZANTINE_CLIENTS,
    BACKDOOR_TARGET_CLASS, DEFAULT_ALPHA, ALPHA_VALUES,
    DEVICE, NUM_DEFENSES,
)
from models import get_model, clone_model, get_flat_params
from data_utils import (
    load_cifar10, dirichlet_partition, build_client_datasets,
    sample_fltrust_root, client_label_distribution, estimate_alpha_mle
)
from client import collect_gradients, per_client_val_loss
from defenses import apply_defense, apply_gradient_to_model, DEFENSE_NAMES
from gia import GIAManager
from state_reward import (
    STATE_DIM, RunningNormaliser, RewardNormaliser,
    build_state_vector, evaluate_accuracy, evaluate_asr, evaluate_loss,
    compute_reward
)
from data_utils import TriggerOnlyDataset, get_test_loader

# Optional lightweight profiling — set DEBUG_TIMING=True in config.py (or
# monkeypatch fl_env.DEBUG_TIMING = True before creating the env) to print
# a per-stage timing breakdown for the first few rounds of an episode, then
# it auto-disables. This exists specifically to diagnose real-hardware
# bottlenecks (e.g. GPU transfer overhead) that don't reproduce in a CPU-only
# sandbox — see project changelog.
try:
    from config import DEBUG_TIMING
except ImportError:
    DEBUG_TIMING = False
_TIMING_ROUNDS_LEFT = [5] if DEBUG_TIMING else [0]

import time as _time
class _Stage:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = _time.time()
        return self
    def __exit__(self, *a):
        if _TIMING_ROUNDS_LEFT[0] > 0:
            print(f"    [TIMING] {self.label}: {_time.time()-self.t0:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# THREAT LABEL LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_threat_label(
    gia_clients_selected: bool,
    backdoor_clients_selected: bool,
    byzantine_clients_selected: bool,
) -> int:
    """
    0=clean, 1=GIA, 2=backdoor, 3=byzantine, 4=mixed
    Mixed = two or more threats simultaneously active.
    """
    active = int(gia_clients_selected) + int(backdoor_clients_selected) + \
             int(byzantine_clients_selected)
    if active == 0:
        return 0
    if active >= 2:
        return 4
    if gia_clients_selected:
        return 1
    if backdoor_clients_selected:
        return 2
    return 3


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class FLDefenseEnv(gym.Env):
    """
    Gymnasium environment for the Threat-Adaptive FL defense selector.

    Parameters
    ──────────
    alpha          : Dirichlet concentration for Non-IID partitioning
    total_rounds   : number of FL rounds per episode
    seed           : random seed for this env instance
    eval_mode      : if True, all three threats are active every round
                     (held-out evaluation config, spec §8 / §9.8)
    smoke_test     : if True, use SMOKE_ROUNDS instead of FULL_ROUNDS
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        total_rounds: int = FULL_ROUNDS,
        seed: int = 42,
        eval_mode: bool = False,
        smoke_test: bool = False,
    ):
        super().__init__()

        self.alpha        = alpha
        self.total_rounds = SMOKE_ROUNDS if smoke_test else total_rounds
        self.base_seed    = seed
        self.eval_mode    = eval_mode

        # ── Action / observation spaces ──────────────────────────────────
        self.action_space      = spaces.Discrete(NUM_DEFENSES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

        # ── Load datasets once (shared across resets) ────────────────────
        self._train_dataset, self._test_dataset = load_cifar10()
        self._trigger_dataset = TriggerOnlyDataset(self._test_dataset)
        self._test_loader    = get_test_loader(self._test_dataset)
        self._trigger_loader = get_test_loader(self._trigger_dataset)

        # FLTrust root dataset — sampled once, never modified
        self._root_dataset = sample_fltrust_root(self._train_dataset, seed=seed)

        # Internal state (initialised in reset())
        self._rng: Optional[np.random.RandomState] = None
        self._round: int = 0
        self._global_model: Optional[torch.nn.Module] = None
        self._client_datasets: Optional[List] = None
        self._gia_manager: Optional[GIAManager] = None
        self._normaliser: Optional[RunningNormaliser] = None
        self._reward_norm: Optional[RewardNormaliser] = None

        
        self._prev_loss: float = 0.0
        self._prev_acc:  float = 0.0
        self._prev_defense: int = 4   # start with NoDefense history

        # Fixed adversary IDs (chosen at reset time)
        self._gia_ids:       List[int] = []
        self._backdoor_ids:  List[int] = []
        self._byzantine_ids: List[int] = []

        # FIX (see project changelog): episode counter used to vary the
        # effective seed across episodes when SB3 auto-resets without
        # passing an explicit seed (see reset() below). A persistent RNG,
        # seeded once here at construction, derives each episode's seed —
        # this is itself deterministic given `seed`, so a full training run
        # is still fully reproducible end-to-end, it just no longer replays
        # the identical 200-round trajectory every episode.
        self._episode_count: int = 0
        self._episode_seed_rng = np.random.RandomState(seed)

    # ── RESET ─────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            # Explicit seed passed (e.g. eval harness) — honour it exactly,
            # and don't let it perturb the episode-seed stream used for
            # ordinary auto-resets.
            effective_seed = seed
        else:
            # FIX (see project changelog): previously this fell back to
            # self.base_seed on EVERY auto-reset (SB3 calls reset() with no
            # seed argument between episodes), so every episode replayed the
            # exact same Non-IID partition, adversary role assignment, and
            # per-round client-sampling stream. Confirmed empirically:
            # episodes 6, 7, and 10 of a real training run were byte-
            # identical (same accuracy, same actions, same threat labels,
            # every round). The only thing that could still change was the
            # PPO policy, so once the policy's entropy collapsed the agent
            # was replaying one fixed trajectory over and over — which also
            # explains the monotonic accuracy decline observed across
            # episodes for that seed (PPO converging to a worse deterministic
            # policy on a static problem, not "not enough episodes yet").
            effective_seed = int(self._episode_seed_rng.randint(0, 2**31 - 1))
            self._episode_count += 1

        self._rng = np.random.RandomState(effective_seed)
        torch.manual_seed(effective_seed)
        random.seed(effective_seed)

        # Fresh global model
        self._global_model = get_model()

        # Fresh Non-IID partitions
        client_indices = dirichlet_partition(
            self._train_dataset,
            num_clients=NUM_CLIENTS,
            alpha=self.alpha,
            seed=effective_seed,
        )

        # Assign adversary roles (fixed for the episode)
        all_ids = list(range(NUM_CLIENTS))
        self._rng.shuffle(all_ids)
        self._gia_ids       = all_ids[:GIA_CLIENTS]
        self._backdoor_ids  = all_ids[GIA_CLIENTS: GIA_CLIENTS + BACKDOOR_CLIENTS]
        self._byzantine_ids = all_ids[GIA_CLIENTS + BACKDOOR_CLIENTS:
                                      GIA_CLIENTS + BACKDOOR_CLIENTS + BYZANTINE_CLIENTS]

        # Build per-client datasets (with backdoor injection)
        self._client_datasets = build_client_datasets(
            self._train_dataset,
            client_indices,
            backdoor_client_ids=self._backdoor_ids,
            seed=effective_seed,
        )

        # Pre-compute label distributions for alpha estimation
        self._label_dists = [
            client_label_distribution(d) for d in self._client_datasets
        ]

        # Reset episode state
        self._round        = 0
        self._prev_loss    = 0.0
        self._prev_acc     = 0.0
        self._prev_defense = 4      # NoDefense as initial "memory"

        self._gia_manager  = GIAManager(device=DEVICE)
        if self._normaliser is None:
            self._normaliser = RunningNormaliser(dim=STATE_DIM)
        if self._reward_norm is None:
            self._reward_norm = RewardNormaliser()
        # Keep normalizer statistics across episodes for stable observations

        # Compute initial accuracy before any training
        self._prev_acc  = evaluate_accuracy(self._global_model, self._test_loader, DEVICE)
        self._prev_loss = evaluate_loss(self._global_model, self._test_loader, DEVICE)

        # Return dummy zero-state (no gradients yet)
        obs = np.zeros(STATE_DIM, dtype=np.float32)
        return obs, {}

    # ── STEP ──────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one FL round with the chosen defense `action`.

        Returns (obs, reward, terminated, truncated, info)
        """
        assert 0 <= action < NUM_DEFENSES, f"Invalid action {action}"

        t = self._round

        # ── 1. Client selection ──────────────────────────────────────────
        selected = self._select_clients(t)

        # Determine which threat types are active this round
        if self.eval_mode:
            # All threats active (held-out config, spec §9.8)
            gia_selected       = True
            backdoor_selected  = True
            byzantine_selected = True
            # Ensure adversary clients are always included in eval mode
            selected = self._ensure_adversaries_selected(selected)
        else:
            gia_selected       = any(c in self._gia_ids       for c in selected)
            backdoor_selected  = any(c in self._backdoor_ids  for c in selected)
            byzantine_selected = any(c in self._byzantine_ids for c in selected)

        threat_label = get_threat_label(
            gia_selected, backdoor_selected, byzantine_selected
        )

        # ── 2. Collect gradients from clients ────────────────────────────
        with _Stage("2. collect_gradients (client local training)"):
            gradients, train_losses, gia_context = collect_gradients(
                global_model=self._global_model,
                selected_client_ids=selected,
                client_datasets=self._client_datasets,
                byzantine_client_ids=self._byzantine_ids,
                gia_client_ids=self._gia_ids,
                device=DEVICE,
            )

        # FIX (see project changelog): if most clients returned a NaN loss
        # this round, the global model going INTO this round was already
        # corrupted — every client trains from the same starting point, so
        # widespread simultaneous NaN losses is a symptom of a bad global
        # model, not 10 independently-unlucky clients. The post-aggregation
        # check further below can miss this (a NaN-safe aggregation can
        # leave params looking finite while still being functionally dead),
        # and waiting for it means burning an entire round's compute on
        # data that gets thrown away anyway. Recover now and redo.
        n_nan_clients = sum(1 for v in train_losses.values() if np.isnan(v))
        if train_losses and n_nan_clients > len(train_losses) / 2:
            print(f"[WARN] Round {t}: {n_nan_clients}/{len(train_losses)} clients "
                  f"returned NaN loss — global model already corrupted entering "
                  f"this round, reinitializing before aggregation.")
            self._global_model = get_model()
            gradients, train_losses, gia_context = collect_gradients(
                global_model=self._global_model,
                selected_client_ids=selected,
                client_datasets=self._client_datasets,
                byzantine_client_ids=self._byzantine_ids,
                gia_client_ids=self._gia_ids,
                device=DEVICE,
            )

        # ── 3. Accuracy BEFORE defense (raw FedAvg, for utility_loss) ───
        acc_before_defense = self._prev_acc
        # ── 4. GIA evaluation (every GIA_EVAL_INTERVAL rounds) ──────────
        # FIX (see project changelog): now passes `action` through so the
        # attack targets the gradient AS TRANSMITTED under the chosen
        # defense, not the raw pre-defense gradient regardless of action.
        # Previously privacy_score was structurally incapable of reflecting
        # the agent's choice at all — confirmed empirically (DP-SGD showed
        # LOWER mean privacy_score than NoDefense across ~7600 pooled
        # rounds). See also client.py's compute_single_step_gradient fix,
        # which this depends on to be attacking a coherent target at all.
        if self._gia_manager.should_evaluate(t):
            with _Stage("4. GIA evaluation (DLG reconstruction)"):
                self._gia_manager.evaluate(t, self._global_model, gia_context, action=action)
        privacy_score = self._gia_manager.get_privacy_score(t)

        # ── 5. Apply chosen defense ──────────────────────────────────────
        with _Stage(f"5. apply_defense (action={action})"):
            aggregated_gradient = apply_defense(
                action=action,
                gradients=gradients,
                global_model=self._global_model,
                root_dataset=self._root_dataset,
                device=DEVICE,
            )
        # Sanitize aggregated gradient before applying to global model
        aggregated_gradient = torch.nan_to_num(
            aggregated_gradient, nan=0.0, posinf=100.0, neginf=-100.0
        )
        aggregated_gradient = torch.clamp(aggregated_gradient, -100.0, 100.0)

        # ── 6. Update global model ───────────────────────────────────────
        # gradient = init_params - trained_params
        # so: new_params = init_params - gradient = trained_params (correct)
        apply_gradient_to_model(self._global_model, aggregated_gradient, lr=1.0)
        # Verify model integrity — reset to fresh model if corrupted
        from models import get_flat_params
        params = get_flat_params(self._global_model)
        param_max = params.abs().max().item()
        # FIX (see project changelog): the original check only caught literal
        # NaN/inf. But apply_gradient_to_model's own nan_to_num clamps
        # posinf/neginf to +-1e4 rather than 0 — so a genuinely blown-up
        # update can leave params "technically finite" at that clamp ceiling,
        # which then overflows to NaN again during the NEXT round's forward
        # pass (repeated matrix multiplication through several layers), with
        # no isnan/isinf ever showing up ON THE PARAMS themselves to trigger
        # recovery. A magnitude threshold catches this before it cascades.
        if torch.isnan(params).any() or torch.isinf(params).any() or param_max > 100.0:
            print(f"[WARN] Round {t}: Model corrupted after defense {action} "
                  f"(max |param|={param_max:.2e}), reinitializing")
            self._global_model = get_model()

        # ── 7. Evaluate updated model ────────────────────────────────────
        with _Stage("7. evaluate_accuracy/asr/loss"):
            curr_acc  = evaluate_accuracy(self._global_model, self._test_loader, DEVICE)
            # Compute loss only every 10 rounds to reduce CUDA memory pressure
            if t % 10 == 0:
                curr_loss = evaluate_loss(self._global_model, self._test_loader, DEVICE)
            else:
                curr_loss = self._prev_loss
            curr_asr  = evaluate_asr(self._global_model, self._trigger_loader,
                                     BACKDOOR_TARGET_CLASS, DEVICE)
        acc_after_defense = curr_acc

        # ── 8. Per-client validation losses (for state) ─────────────────
        client_val_losses = {cid: train_losses.get(cid, 0.0) 
                            for cid in selected}

        # ── 9. Alpha MLE estimate ────────────────────────────────────────
        alpha_hat = estimate_alpha_mle(self._label_dists)

        # ── 10. Build state vector ───────────────────────────────────────
        raw_state = build_state_vector(
            gradients=gradients,
            prev_loss=self._prev_loss,
            curr_loss=curr_loss,
            prev_acc=self._prev_acc,
            curr_acc=curr_acc,
            client_val_losses=client_val_losses,
            alpha_estimate=alpha_hat,
            round_idx=t,
            total_rounds=self.total_rounds,
            prev_defense=self._prev_defense,
        )
        obs = self._normaliser.update_and_normalise(raw_state)

        # ── 11. Compute reward ───────────────────────────────────────────
        raw_reward = compute_reward(
            action=action,
            prev_acc=self._prev_acc,
            curr_acc=curr_acc,
            privacy_score=privacy_score,
            acc_before_defense=acc_before_defense,
            acc_after_defense=acc_after_defense,
        )
        if np.isnan(raw_reward) or np.isinf(raw_reward):
            raw_reward = 0.0
        reward = self._reward_norm.update_and_normalise(raw_reward)

        # ── 12. Advance state ────────────────────────────────────────────
        self._prev_loss    = curr_loss
        self._prev_acc     = curr_acc
        self._prev_defense = action
        self._round       += 1

        terminated = False
        truncated  = self._round >= self.total_rounds

        if _TIMING_ROUNDS_LEFT[0] > 0:
            _TIMING_ROUNDS_LEFT[0] -= 1
            if _TIMING_ROUNDS_LEFT[0] == 0:
                print("    [TIMING] profiling window ended (5 rounds shown above)")

        info = {
            "round":          t,
            "acc":            curr_acc,
            "asr":            curr_asr,
            "gia_loss":       1.0 - privacy_score,   # normalised MSE
            "threat_label":   threat_label,
            "defense_chosen": action,
            "defense_name":   DEFENSE_NAMES[action],
            "raw_reward":     raw_reward,
            "privacy_score":  privacy_score,
            "train_loss_mean": float(np.mean(list(train_losses.values()))),
        }

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    # ── HELPERS ───────────────────────────────────────────────────────────

    def _select_clients(self, round_idx: int) -> List[int]:
        """Select CLIENTS_PER_ROUND clients uniformly at random."""
        idxs = list(range(NUM_CLIENTS))
        self._rng.shuffle(idxs)
        return idxs[:CLIENTS_PER_ROUND]

    def _ensure_adversaries_selected(self, selected: List[int]) -> List[int]:
        """For eval mode: ensure at least one of each adversary type is included."""
        selected_set = set(selected)
        to_add = []
        for adv_list in [self._gia_ids, self._backdoor_ids, self._byzantine_ids]:
            if not any(a in selected_set for a in adv_list):
                to_add.append(adv_list[0])
        if to_add:
            # Replace last N normal clients with adversaries
            normal = [c for c in selected if c not in
                      set(self._gia_ids + self._backdoor_ids + self._byzantine_ids)]
            for i, adv in enumerate(to_add[:len(normal)]):
                normal[-(i + 1)] = adv
            selected = normal + [c for c in selected if c in
                                 set(self._gia_ids + self._backdoor_ids + self._byzantine_ids)]
            selected = list(dict.fromkeys(selected))[:CLIENTS_PER_ROUND]
        return selected

    def render(self):
        pass   # no rendering needed


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: raw FedAvg (for utility_loss computation)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_fedavg(gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
    mat = torch.stack(list(gradients.values()), dim=0)
    return mat.mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke test (2 rounds) …")
    env = FLDefenseEnv(alpha=0.5, smoke_test=True, seed=0)
    obs, info = env.reset(seed=0)
    print(f"  Initial obs shape: {obs.shape}  (expected {STATE_DIM})")

    for step in range(2):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Round {info['round']:3d} | "
              f"Defense: {info['defense_name']:10s} | "
              f"Acc: {info['acc']:.4f} | "
              f"ASR: {info['asr']:.4f} | "
              f"Reward: {reward:.4f}")

    print("\n✓ fl_env smoke test passed.")