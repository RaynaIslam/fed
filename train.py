"""
train.py — PPO agent training over the FL defense environment.

Usage
─────
# Smoke test (CPU, tiny config):
    python train.py --smoke --seed 0

# Single full run:
    python train.py --alpha 0.1 --seed 0

# All seeds × all alpha values (full paper results):
    python train.py --full
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from config import (
    RESULTS_DIR, MODELS_DIR,
    NUM_SEEDS, ALPHA_VALUES, DEFAULT_ALPHA,
    FULL_ROUNDS, SMOKE_ROUNDS,
    PPO_LR_START, PPO_LR_END,
    PPO_CLIP_EPS, PPO_ENT_COEF, PPO_VF_COEF,
    PPO_GAE_LAMBDA, PPO_GAMMA,
    PPO_BATCH_SIZE, PPO_N_EPOCHS, PPO_N_STEPS,
    PPO_N_ENVS, POLICY_NET_ARCH,
    PPO_TOTAL_EPISODES, SMOKE_EPISODES,
    CHECKPOINT_EVERY, DEVICE,
)
from fl_env import FLDefenseEnv
from state_reward import STATE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def linear_lr_schedule(lr_start: float, lr_end: float):
    """
    Returns a callable that SB3 uses to anneal the learning rate.
    progress_remaining goes from 1.0 → 0.0 over training.
    """
    def schedule(progress_remaining: float) -> float:
        return lr_end + progress_remaining * (lr_start - lr_end)
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY  (resume support)
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(model_prefix: str) -> Optional[str]:
    """
    Return the path to the highest-round checkpoint for this (alpha, seed)
    run, or None if none exist.

    Parses the round number out of the filename ({model_prefix}_ckptN.zip)
    rather than sorting lexicographically — a naive string sort would rank
    "ckpt1400" before "ckpt700".
    """
    candidates = glob.glob(f"{model_prefix}_ckpt*.zip")
    if not candidates:
        return None
    best_path, best_round = None, -1
    for path in candidates:
        m = re.search(r"_ckpt(\d+)\.zip$", path)
        if m and int(m.group(1)) > best_round:
            best_round = int(m.group(1))
            best_path = path
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class FLMetricsCallback(BaseCallback):
    """
    Collects per-round metrics from the info dicts returned by env.step()
    and writes them to a CSV file.

    Works with both DummyVecEnv and SubprocVecEnv.
    """

    def __init__(
        self,
        csv_path: str,
        total_rounds: int,
        checkpoint_every: int = CHECKPOINT_EVERY,
        model_save_prefix: str = "",
        verbose: int = 0,
        resume: bool = False,
    ):
        super().__init__(verbose)
        self.csv_path         = csv_path
        self.total_rounds     = total_rounds
        self.checkpoint_every = checkpoint_every
        self.model_save_prefix = model_save_prefix
        self.resume           = resume
        self._csv_file   = None
        self._csv_writer = None
        self._round_count = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        fieldnames=[
            "episode", "round", "acc", "asr", "gia_loss", "threat_label",
            "defense_chosen", "defense_name", "raw_reward",
            "privacy_score", "train_loss_mean",
        ]
        existing_rows = []
        file_has_rows = self.resume and os.path.exists(self.csv_path) \
            and os.path.getsize(self.csv_path) > 0
        if file_has_rows:
            with open(self.csv_path, "r", newline="") as f:
                existing_rows = list(csv.DictReader(f))
            file_has_rows = len(existing_rows) > 0

        self._csv_file = open(self.csv_path, "a" if file_has_rows else "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)

        if file_has_rows:
            # Resume: pick up episode/round counters from the last row
            # already on disk instead of restarting at (episode=0, round=0)
            # and corrupting the episode numbering for the rest of the run.
            last = existing_rows[-1]
            self._episode_idx = int(last["episode"])
            self._prev_round  = int(last["round"])
            self._round_count = len(existing_rows)
        else:
            self._csv_writer.writeheader()
            self._episode_idx = 0
            self._prev_round = None


    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not info:
                continue
            round_num = info.get("round", 0)

            # FIX (see project changelog): the original code had
            #   if round_num == 0 and self._round_count > 0: break
            # (duplicated twice) which SKIPPED logging round 0 of every
            # episode after the first — silently dropping one row per
            # episode (9 missing rows across a 10-episode run) instead of
            # just tracking which episode a row belongs to. We now log
            # every row unconditionally and instead track episode
            # boundaries explicitly via an `episode` column, using the
            # same "round decreased" signal — this also makes each row
            # traceable to its episode for later analysis, which the
            # original schema didn't support at all.
            if self._prev_round is not None and round_num < self._prev_round:
                self._episode_idx += 1
            self._prev_round = round_num

            row = {
                "episode":         self._episode_idx,
                "round":           round_num,
                "acc":             info.get("acc", 0.0),
                "asr":             info.get("asr", 0.0),
                "gia_loss":        info.get("gia_loss", 0.0),
                "threat_label":    info.get("threat_label", 0),
                "defense_chosen":  info.get("defense_chosen", -1),
                "defense_name":    info.get("defense_name", ""),
                "raw_reward":      info.get("raw_reward", 0.0),
                "privacy_score":   info.get("privacy_score", 0.0),
                "train_loss_mean": info.get("train_loss_mean", 0.0),
            }
            self._csv_writer.writerow(row)
            self._round_count += 1

            if (self._round_count % self.checkpoint_every == 0
                    and self.model_save_prefix):
                # Same pathlib-suffix issue as the final-model save — see
                # train_one_run()'s comment. Explicit .zip required.
                ckpt_path = f"{self.model_save_prefix}_ckpt{self._round_count}.zip"
                self.model.save(ckpt_path)

        if self._csv_file:
            self._csv_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env(alpha: float, seed: int, total_rounds: int, smoke: bool):
    """Returns a callable that creates one FLDefenseEnv (required by SB3)."""
    def _init():
        set_random_seed(seed)
        env = FLDefenseEnv(
            alpha=alpha,
            total_rounds=total_rounds,
            seed=seed,
            eval_mode=False,
            smoke_test=smoke,
        )
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(
    alpha: float,
    seed: int,
    smoke: bool = False,
    n_envs: int = PPO_N_ENVS,
    verbose: int = 1,
) -> str:
    """
    Train one PPO run (one seed, one alpha).
    Returns path to saved model zip.
    """
    total_rounds = SMOKE_ROUNDS if smoke else FULL_ROUNDS
    run_tag      = f"alpha{alpha}_seed{seed}{'_smoke' if smoke else ''}"
    model_prefix = os.path.join(MODELS_DIR,  f"ppo_agent_{run_tag}")
    final_path   = f"{model_prefix}_final.zip"
    csv_path     = os.path.join(RESULTS_DIR, f"metrics_{run_tag}.csv")

    # FIX (Bug 12, see project changelog): if this (alpha, seed) run already
    # completed, don't retrain from scratch — skip cleanly.
    if os.path.exists(final_path):
        print(f"\n  ✓ {run_tag} already complete ({final_path}) — skipping.")
        return final_path

    print(f"\n{'='*60}")
    print(f"  Training run: {run_tag}")
    print(f"  Alpha={alpha}  Seed={seed}  Rounds={total_rounds}  "
          f"Envs={n_envs}  Device={DEVICE}")
    print(f"{'='*60}")

    # ── Vectorised environments ──────────────────────────────────────────
    # Use DummyVecEnv on Windows for smoke tests (SubprocVecEnv has
    # spawn overhead and pickling requirements on Windows).
    # Use SubprocVecEnv for full runs on Linux/Windows with proper guards.
    use_subproc = (not smoke) and (n_envs > 1) and (os.name != "nt")

    env_fns = [make_env(alpha, seed + i, total_rounds, smoke)
               for i in range(n_envs)]

    actual_n_envs = 1
    env_fns = env_fns[:actual_n_envs]
    vec_env = DummyVecEnv(env_fns)

    # ── PPO agent — fresh, or resumed from the latest checkpoint ──────────
    # FIX (Bug 12, see project changelog): checkpoints were saved but never
    # loaded back, so every interruption meant restarting at episode 0. This
    # resumes the PPO policy's learned weights and cumulative timestep count
    # (not mid-episode FL environment state — that isn't part of an SB3
    # checkpoint and would be fragile to serialize; the agent starts a fresh
    # episode going forward rather than replaying a partial one).
    n_episodes = SMOKE_EPISODES if smoke else PPO_TOTAL_EPISODES
    total_timesteps = total_rounds * n_episodes

    ckpt_path = find_latest_checkpoint(model_prefix)
    resume = ckpt_path is not None

    if resume:
        print(f"  Resuming from checkpoint: {ckpt_path}")
        model = PPO.load(ckpt_path, env=vec_env, device="cpu")
        remaining_timesteps = max(total_timesteps - model.num_timesteps, 0)
        if remaining_timesteps == 0:
            print(f"  Checkpoint already covers all {total_timesteps} "
                  f"timesteps — saving final model without further training.")
            model.save(final_path)
            vec_env.close()
            return final_path
        print(f"  Already trained {model.num_timesteps}/{total_timesteps} "
              f"timesteps — training remaining {remaining_timesteps}.")
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=POLICY_NET_ARCH, vf=POLICY_NET_ARCH),
            activation_fn=torch.nn.ReLU,
        )
        lr_schedule = linear_lr_schedule(PPO_LR_START, PPO_LR_END)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_EPS,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            verbose=verbose,
            seed=seed,
            device="cpu",   # PPO MLP is tiny — CPU is fine; GPU used inside FL env
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
        )
        remaining_timesteps = total_timesteps
    # ── Callbacks ────────────────────────────────────────────────────────
    metrics_callback = FLMetricsCallback(
        csv_path=csv_path,
        total_rounds=total_rounds,
        checkpoint_every=CHECKPOINT_EVERY,
        model_save_prefix=model_prefix,
        verbose=verbose,
        resume=resume,
    )

    # ── Train ────────────────────────────────────────────────────────────
    
    t0 = time.time()
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=metrics_callback,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )
    elapsed = time.time() - t0

    # ── Save final model ─────────────────────────────────────────────────
    # FIX (see project changelog): SB3 only auto-appends .zip when
    # pathlib.Path(save_path).suffix == "". Because model_prefix embeds the
    # raw alpha float (e.g. "alpha0.1"), pathlib sees the tail of "0.1" as
    # an existing suffix and skips the auto-append — for EVERY alpha value
    # in ALPHA_VALUES, since they all contain a decimal point. Confirmed
    # empirically: models were being saved with no extension at all, while
    # evaluate.py's glob pattern requires .zip, so full_evaluation() would
    # silently find zero PPO models. Appending .zip explicitly here sidesteps
    # pathlib's suffix detection entirely (path already ends in .zip, so
    # nothing further gets appended — verified no double-extension results).
    model.save(final_path)
    vec_env.close()

    print(f"\n  ✓ Run complete in {elapsed/60:.1f} min")
    print(f"  Model saved : {final_path}")
    print(f"  Metrics CSV : {csv_path}")

    return final_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO defense selector for Federated Learning"
    )
    p.add_argument("--smoke",  action="store_true",
                   help="Smoke test: tiny config, CPU, 5 rounds")
    p.add_argument("--full",   action="store_true",
                   help="Full run: all seeds × all alpha values")
    p.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA,
                   help=f"Dirichlet alpha (default {DEFAULT_ALPHA})")
    p.add_argument("--seed",   type=int, default=0,
                   help="Random seed (default 0)")
    p.add_argument("--n_envs", type=int, default=PPO_N_ENVS,
                   help=f"Parallel envs (default {PPO_N_ENVS}; forced=1 on smoke)")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    # Required for SubprocVecEnv on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()

    # Global seeds
    torch.manual_seed(42)
    np.random.seed(42)

    if args.smoke:
        print("\n" + "="*60)
        print("  SMOKE TEST MODE")
        print(f"  Rounds : {SMOKE_ROUNDS}")
        print(f"  Device : CPU (forced for smoke test)")
        print(f"  Envs   : 1 (DummyVecEnv)")
        print("="*60)
        train_one_run(
            alpha=0.5,
            seed=0,
            smoke=True,
            n_envs=1,
            verbose=args.verbose,
        )

    elif args.full:
        print("\nFULL TRAINING — all seeds × all alpha values")
        for alpha in ALPHA_VALUES:
            for seed in range(NUM_SEEDS):
                train_one_run(
                    alpha=alpha,
                    seed=seed,
                    smoke=False,
                    n_envs=args.n_envs,
                    verbose=args.verbose,
                )

    else:
        # Single run
        train_one_run(
            alpha=args.alpha,
            seed=args.seed,
            smoke=False,
            n_envs=args.n_envs,
            verbose=args.verbose,
        )
