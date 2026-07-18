"""
baselines.py — Evaluate all 5 baseline defense policies under identical FL
               settings as the PPO agent.

Baselines (spec §7):
    0: Always-DP-SGD   — fixed action 0 every round
    1: Always-FLTrust  — fixed action 1 every round
    2: Always-Krum     — fixed action 2 every round
    3: Always-FLAME    — fixed action 3 every round
    4: Random          — uniform random from {0,1,2,3,4} each round

Usage
─────
    python baselines.py --smoke                  # quick test
    python baselines.py --alpha 0.1 --seed 0     # single run
    python baselines.py --full                   # all alpha × seeds
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from config import (
    RESULTS_DIR, NUM_SEEDS, ALPHA_VALUES, DEFAULT_ALPHA,
    FULL_ROUNDS, SMOKE_ROUNDS, NUM_DEFENSES, DEVICE,
)
from fl_env import FLDefenseEnv
from defenses import DEFENSE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# POLICY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_POLICIES = {
    "always_dp_sgd":  lambda obs, rng: 0,
    "always_fltrust": lambda obs, rng: 1,
    "always_krum":    lambda obs, rng: 2,
    "always_flame":   lambda obs, rng: 3,
    "random":         lambda obs, rng: int(rng.integers(0, NUM_DEFENSES)),
}


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    policy_name: str,
    alpha: float,
    seed: int,
    smoke: bool = False,
    verbose: bool = True,
    eval_mode: bool = False,
    total_rounds: int | None = None,
) -> str:
    """
    Run one baseline policy for a full episode and write metrics to CSV.
    Returns the path to the CSV file.

    eval_mode / total_rounds: FIX (see project changelog) — added so this
    function can be run under the SAME held-out condition used for PPO
    evaluation (evaluate.py's evaluate_ppo(): eval_mode=True, seed+1000,
    EVAL_ROUNDS). Defaults preserve the original standalone behavior of
    this script (ordinary training-distribution condition, FULL_ROUNDS),
    so `python baselines.py --full` etc. are unaffected.
    """
    total_rounds = total_rounds if total_rounds is not None else (
        SMOKE_ROUNDS if smoke else FULL_ROUNDS
    )
    suffix = "_smoke" if smoke else ("_eval" if eval_mode else "")
    run_tag  = f"{policy_name}_alpha{alpha}_seed{seed}{suffix}"
    csv_path = os.path.join(RESULTS_DIR, f"metrics_{run_tag}.csv")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    policy_fn = BASELINE_POLICIES[policy_name]
    rng       = np.random.default_rng(seed)

    env = FLDefenseEnv(
        alpha=alpha,
        total_rounds=total_rounds,
        seed=seed,
        eval_mode=eval_mode,
        smoke_test=smoke,
    )

    obs, _ = env.reset(seed=seed)

    if verbose:
        print(f"\n  Baseline: {policy_name:20s} | alpha={alpha} | seed={seed} | "
              f"rounds={total_rounds}")

    rows = []
    t0   = time.time()

    for _ in range(total_rounds):
        action = policy_fn(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)

        rows.append({
            "round":           info.get("round", 0),
            "acc":             info.get("acc", 0.0),
            "asr":             info.get("asr", 0.0),
            "gia_loss":        info.get("gia_loss", 0.0),
            "threat_label":    info.get("threat_label", 0),
            "defense_chosen":  info.get("defense_chosen", action),
            "defense_name":    info.get("defense_name", DEFENSE_NAMES.get(action, "")),
            "raw_reward":      info.get("raw_reward", 0.0),
            "privacy_score":   info.get("privacy_score", 0.0),
            "train_loss_mean": info.get("train_loss_mean", 0.0),
        })

        if verbose and (info.get("round", 0) % 10 == 0 or smoke):
            print(f"    Round {info['round']:3d} | "
                  f"Acc={info['acc']:.4f} | "
                  f"ASR={info['asr']:.4f} | "
                  f"Privacy={info['privacy_score']:.4f} | "
                  f"Defense={info['defense_name']}")

        if terminated or truncated:
            break

    elapsed = time.time() - t0

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        last = rows[-1]
        print(f"  Done in {elapsed/60:.1f} min | "
              f"Final Acc={last['acc']:.4f} | "
              f"Final ASR={last['asr']:.4f} | "
              f"CSV: {csv_path}")

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    acc: float,
    privacy_score: float,
    asr: float,
) -> float:
    """
    spec §7: 0.4*final_acc + 0.3*avg_privacy_score + 0.3*(1 - avg_asr)
    averaged over last 50 rounds of evaluation.
    """
    from config import COMPOSITE_W_ACC, COMPOSITE_W_PRIVACY, COMPOSITE_W_ASR
    return (
        COMPOSITE_W_ACC     * acc
        + COMPOSITE_W_PRIVACY * privacy_score
        + COMPOSITE_W_ASR     * (1.0 - asr)
    )


def compute_composite_from_csv(csv_path: str, last_n: int = 50) -> dict:
    """Read a metrics CSV and compute composite score over the last N rounds."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    tail = df.tail(last_n)
    acc     = tail["acc"].mean()
    privacy = tail["privacy_score"].mean()
    asr     = tail["asr"].mean()
    score   = composite_score(acc, privacy, asr)
    return {
        "mean_acc":     acc,
        "mean_privacy": privacy,
        "mean_asr":     asr,
        "composite":    score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run FL defense baselines")
    p.add_argument("--smoke",  action="store_true",
                   help="Smoke test mode (5 rounds, fast)")
    p.add_argument("--full",   action="store_true",
                   help="All seeds × all alpha values")
    p.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--policy", type=str,   default="all",
                   choices=list(BASELINE_POLICIES.keys()) + ["all"])
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    policies_to_run = (list(BASELINE_POLICIES.keys())
                       if args.policy == "all" else [args.policy])

    if args.smoke:
        print("\n" + "="*60)
        print("  BASELINE SMOKE TEST")
        print(f"  Rounds : {SMOKE_ROUNDS}  |  Alpha: 0.5  |  Seed: 0")
        print("="*60)
        for pol in policies_to_run:
            run_baseline(pol, alpha=0.5, seed=0, smoke=True)

    elif args.full:
        summary_rows = []
        for alpha in ALPHA_VALUES:
            for seed in range(NUM_SEEDS):
                for pol in policies_to_run:
                    csv_path = run_baseline(pol, alpha=alpha, seed=seed, smoke=False)
                    scores   = compute_composite_from_csv(csv_path)
                    summary_rows.append({
                        "policy": pol,
                        "alpha":  alpha,
                        "seed":   seed,
                        **scores,
                    })

        # Save summary
        import pandas as pd
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, "baseline_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Baseline summary saved to {summary_path}")
        print(summary_df.groupby(["policy", "alpha"])["composite"].mean().round(4))

    else:
        for pol in policies_to_run:
            run_baseline(pol, alpha=args.alpha, seed=args.seed, smoke=args.smoke)