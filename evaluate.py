"""
evaluate.py — FIXED post-training evaluation ensuring PPO and baselines
               are compared under IDENTICAL conditions.

Bug fixed (2025-06):
    The original full_evaluation() called run_baseline() with:
        eval_mode=False   — baselines saw ordinary training threat distribution
        total_rounds=200  — baselines ran 200 rounds; only tail(50) was compared
        seed=seed         — baselines used training seed, not held-out seed
    While evaluate_ppo() used:
        eval_mode=True    — all 3 threats forced active
        total_rounds=50   — only 50 rounds total
        seed=seed+1000    — held-out seed never seen in training

    This made the comparison unfair in BOTH directions:
      • Baselines ran on an easier threat distribution (eval_mode=False)
      • Baselines ran 200 rounds and were scored on the last 50 (mature model)
      • PPO was scored on rounds 0-49 of a fresh hard environment

Fix: run_baseline_eval() mirrors evaluate_ppo() exactly:
    eval_mode=True, total_rounds=EVAL_ROUNDS, seed=seed+1000

Usage
─────
    python evaluate.py --model models/ppo_agent_alpha0.1_seed0_final.zip
    python evaluate.py --full
    python evaluate.py --full --smoke
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from config import (
    RESULTS_DIR, MODELS_DIR, EVAL_ROUNDS, ALPHA_VALUES,
    DEFAULT_ALPHA, SMOKE_ROUNDS, NUM_DEFENSES, DEVICE,
)
from fl_env import FLDefenseEnv
from defenses import DEFENSE_NAMES
from baselines import BASELINE_POLICIES, composite_score


# ─────────────────────────────────────────────────────────────────────────────
# SHARED EVALUATION ENVIRONMENT FACTORY
# Creates identical environments for PPO and baselines
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_env(
    alpha: float,
    seed: int,
    smoke: bool = False,
) -> FLDefenseEnv:
    """
    Single factory used by BOTH evaluate_ppo() and run_baseline_eval().
    Guarantees identical conditions:
        eval_mode   = True          all 3 threats active every round
        total_rounds = EVAL_ROUNDS  50 rounds (or SMOKE_ROUNDS if smoke)
        seed        = seed + 1000   never seen during training
    """
    total_rounds = min(SMOKE_ROUNDS, EVAL_ROUNDS) if smoke else EVAL_ROUNDS
    return FLDefenseEnv(
        alpha=alpha,
        total_rounds=total_rounds,
        seed=seed + 1000,   # held-out: never seen during training
        eval_mode=True,     # all 3 threats forced active simultaneously
        smoke_test=smoke,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE PPO AGENT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ppo(
    model_path: str,
    alpha: float,
    seed: int,
    smoke: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a trained PPO model and run it deterministically on the held-out
    eval environment (all threats active simultaneously).
    Returns a DataFrame with per-round metrics.
    """
    env   = make_eval_env(alpha, seed, smoke)
    total = env.total_rounds
    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset(seed=seed + 1000)

    rows = []
    defense_counts = {i: 0 for i in range(NUM_DEFENSES)}

    if verbose:
        print(f"\n  [PPO] Evaluating: {os.path.basename(model_path)}"
              f" | alpha={alpha} | seed+1000={seed+1000} | rounds={total}"
              f" | eval_mode=True")

    for _ in range(total):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        defense_counts[action] += 1

        rows.append({
            "round":          info.get("round", 0),
            "acc":            info.get("acc", 0.0),
            "asr":            info.get("asr", 0.0),
            "gia_loss":       info.get("gia_loss", 0.0),
            "privacy_score":  info.get("privacy_score", 0.0),
            "defense_chosen": action,
            "defense_name":   DEFENSE_NAMES[action],
            "raw_reward":     info.get("raw_reward", 0.0),
            "threat_label":   info.get("threat_label", 4),
        })

        if verbose and (info.get("round", 0) % 10 == 0 or smoke):
            print(f"    Round {info['round']:3d} | "
                  f"Acc={info['acc']:.4f} | "
                  f"ASR={info['asr']:.4f} | "
                  f"Privacy={info['privacy_score']:.4f} | "
                  f"Defense={info['defense_name']}")

        if terminated or truncated:
            break

    df = pd.DataFrame(rows)

    if verbose:
        tail = df.tail(len(df))   # all rounds (already EVAL_ROUNDS)
        comp = composite_score(tail["acc"].mean(), tail["privacy_score"].mean(),
                               tail["asr"].mean())
        print(f"\n  [PPO] Summary (all {len(df)} eval rounds):")
        print(f"    Accuracy  : {tail['acc'].mean():.4f} ± {tail['acc'].std():.4f}")
        print(f"    ASR       : {tail['asr'].mean():.4f} ± {tail['asr'].std():.4f}")
        print(f"    Privacy   : {tail['privacy_score'].mean():.4f}")
        print(f"    Composite : {comp:.4f}")
        print(f"    Defenses  : "
              + ", ".join(f"{DEFENSE_NAMES[k]}={v}"
                          for k, v in defense_counts.items() if v > 0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE BASELINE — mirrors evaluate_ppo() exactly
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_eval(
    policy_name: str,
    alpha: float,
    seed: int,
    smoke: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run a baseline policy under IDENTICAL conditions to evaluate_ppo():
        eval_mode=True, total_rounds=EVAL_ROUNDS, seed=seed+1000

    This is the ONLY correct way to compare baselines against PPO.
    Do NOT use run_baseline() from baselines.py for comparison —
    that function uses eval_mode=False and full 200 training rounds.
    """
    env   = make_eval_env(alpha, seed, smoke)
    total = env.total_rounds
    policy_fn = BASELINE_POLICIES[policy_name]
    rng = np.random.default_rng(seed + 1000)   # same offset as eval env seed

    obs, _ = env.reset(seed=seed + 1000)

    if verbose:
        print(f"\n  [BL] {policy_name:20s} | alpha={alpha} | "
              f"seed+1000={seed+1000} | rounds={total} | eval_mode=True")

    rows = []
    t0 = time.time()

    for _ in range(total):
        action = policy_fn(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)

        rows.append({
            "round":          info.get("round", 0),
            "acc":            info.get("acc", 0.0),
            "asr":            info.get("asr", 0.0),
            "gia_loss":       info.get("gia_loss", 0.0),
            "privacy_score":  info.get("privacy_score", 0.0),
            "defense_chosen": info.get("defense_chosen", action),
            "defense_name":   info.get("defense_name",
                                       DEFENSE_NAMES.get(action, "")),
            "raw_reward":     info.get("raw_reward", 0.0),
            "threat_label":   info.get("threat_label", 4),
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(rows)

    if verbose:
        comp = composite_score(df["acc"].mean(), df["privacy_score"].mean(),
                               df["asr"].mean())
        print(f"    Acc={df['acc'].mean():.4f} | "
              f"ASR={df['asr'].mean():.4f} | "
              f"Composite={comp:.4f} | "
              f"Time={time.time()-t0:.0f}s")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE SPEED
# ─────────────────────────────────────────────────────────────────────────────

def rounds_to_target(df: pd.DataFrame, target: float = 0.60) -> int:
    hits = df[df["acc"] >= target]
    return int(hits.iloc[0]["round"]) if not hits.empty else -1


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION — FAIR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(smoke: bool = False) -> pd.DataFrame:
    """
    Evaluate ALL saved PPO models + ALL baseline policies under IDENTICAL
    conditions and write results/summary_table.csv.

    Both PPO and baselines use:
        make_eval_env() → eval_mode=True, EVAL_ROUNDS, seed+1000
    """
    summary_rows = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_seeds = 1 if smoke else 5

    for alpha in ALPHA_VALUES:
        print(f"\n{'='*60}")
        print(f"  alpha = {alpha}")
        print(f"{'='*60}")

        for seed in range(n_seeds):

            # ── PPO ────────────────────────────────────────────────────
            # FIX (see project changelog): train.py previously saved model
            # files with NO extension at all — pathlib's suffix detection
            # gets confused by the decimal point in "alpha0.1" and SB3 skips
            # its auto .zip-append for every alpha value you have. That's
            # fixed in train.py now, but check both patterns here so any
            # models already trained under the old broken naming still get
            # picked up rather than silently skipped.
            model_pattern = os.path.join(
                MODELS_DIR, f"ppo_agent_alpha{alpha}_seed{seed}_final.zip"
            )
            matching = glob.glob(model_pattern)
            if not matching:
                legacy_pattern = os.path.join(
                    MODELS_DIR, f"ppo_agent_alpha{alpha}_seed{seed}_final"
                )
                matching = glob.glob(legacy_pattern)
                if matching and matching[0] not in ("", None):
                    print(f"  [NOTE] Found model under legacy (no-extension) "
                          f"naming for alpha={alpha} seed={seed} — consider "
                          f"retraining once train.py's save-path fix is in "
                          f"place, so future runs use the correct naming.")

            if matching:
                df_ppo = evaluate_ppo(
                    model_path=matching[0],
                    alpha=alpha, seed=seed,
                    smoke=smoke, verbose=True,
                )
                # Save per-run eval CSV
                eval_csv = os.path.join(
                    RESULTS_DIR, f"eval_ppo_alpha{alpha}_seed{seed}.csv"
                )
                df_ppo.to_csv(eval_csv, index=False)

                summary_rows.append(_build_row(
                    method="PPO", alpha=alpha, seed=seed, df=df_ppo,
                    note="eval_mode=True, seed+1000, EVAL_ROUNDS"
                ))
            else:
                print(f"  [WARN] No PPO model: alpha={alpha} seed={seed}")

            # ── Baselines — identical eval conditions ───────────────────
            for pol_name in BASELINE_POLICIES:
                df_bl = run_baseline_eval(
                    policy_name=pol_name,
                    alpha=alpha, seed=seed,
                    smoke=smoke, verbose=True,
                )
                # Save per-baseline eval CSV
                bl_csv = os.path.join(
                    RESULTS_DIR,
                    f"eval_{pol_name}_alpha{alpha}_seed{seed}.csv"
                )
                df_bl.to_csv(bl_csv, index=False)

                summary_rows.append(_build_row(
                    method=pol_name, alpha=alpha, seed=seed, df=df_bl,
                    note="eval_mode=True, seed+1000, EVAL_ROUNDS"
                ))

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    summary_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Fair comparison summary saved to {out_path}")
    print(f"{'='*60}")
    print("\nComposite scores (mean over seeds, eval_mode=True):")
    pivot = (summary_df
             .groupby(["method", "alpha"])["composite"]
             .mean()
             .round(4)
             .unstack("alpha"))
    print(pivot.to_string())

    return summary_df


def _build_row(method, alpha, seed, df, note=""):
    """Build a summary row from an eval DataFrame."""
    acc  = df["acc"].mean()
    asr  = df["asr"].mean()
    priv = df["privacy_score"].mean()
    comp = composite_score(acc, priv, asr)
    return {
        "method":       method,
        "alpha":        alpha,
        "seed":         seed,
        "n_rounds":     len(df),
        "mean_acc":     round(acc, 4),
        "std_acc":      round(df["acc"].std(), 4),
        "mean_asr":     round(asr, 4),
        "std_asr":      round(df["asr"].std(), 4),
        "mean_privacy": round(priv, 4),
        "composite":    round(comp, 4),
        "rounds_to_50": rounds_to_target(df, 0.50),
        "rounds_to_60": rounds_to_target(df, 0.60),
        "eval_note":    note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK — run this first to verify conditions are identical
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(alpha: float = 0.1, seed: int = 0):
    """
    Confirm PPO and baselines use identical environments.
    Run before full_evaluation().

    NOTE (see project changelog): the original version of this check only
    compared .alpha/.total_rounds/.eval_mode on two freshly-CONSTRUCTED
    (never reset()) env instances — since both come from the same
    make_eval_env() call with the same arguments, those three attributes
    are trivially equal by construction and would pass even if the
    underlying reset()/seeding logic were broken. This version resets both
    envs and compares the actually-realized adversary role assignment and
    a few rounds of client sampling — the property that actually matters.
    """
    print("\n=== SANITY CHECK: Environment conditions ===")
    env_ppo = make_eval_env(alpha, seed, smoke=False)
    env_bl  = make_eval_env(alpha, seed, smoke=False)

    print(f"PPO env:      alpha={env_ppo.alpha} | total_rounds={env_ppo.total_rounds}"
          f" | eval_mode={env_ppo.eval_mode} | seed={seed+1000}")
    print(f"Baseline env: alpha={env_bl.alpha}  | total_rounds={env_bl.total_rounds}"
          f" | eval_mode={env_bl.eval_mode} | seed={seed+1000}")

    assert env_ppo.alpha        == env_bl.alpha,        "alpha mismatch"
    assert env_ppo.total_rounds == env_bl.total_rounds, "total_rounds mismatch"
    assert env_ppo.eval_mode    == env_bl.eval_mode,    "eval_mode mismatch"

    env_ppo.reset(seed=seed + 1000)
    env_bl.reset(seed=seed + 1000)

    assert env_ppo._gia_ids       == env_bl._gia_ids,       \
        f"gia_ids mismatch: {env_ppo._gia_ids} vs {env_bl._gia_ids}"
    assert env_ppo._backdoor_ids  == env_bl._backdoor_ids,  \
        f"backdoor_ids mismatch: {env_ppo._backdoor_ids} vs {env_bl._backdoor_ids}"
    assert env_ppo._byzantine_ids == env_bl._byzantine_ids, \
        f"byzantine_ids mismatch: {env_ppo._byzantine_ids} vs {env_bl._byzantine_ids}"

    n_check = min(5, env_ppo.total_rounds)
    for r in range(n_check):
        sel_ppo = env_ppo._select_clients(r)
        sel_bl  = env_bl._select_clients(r)
        assert sel_ppo == sel_bl, \
            f"round {r} client selection mismatch: {sel_ppo} vs {sel_bl}"

    print(f"✓ Adversary roles match: gia={env_ppo._gia_ids} "
          f"backdoor={env_ppo._backdoor_ids} byzantine={env_ppo._byzantine_ids}")
    print(f"✓ Client sampling matches for {n_check} rounds checked.")
    print("✓ Environments are genuinely identical — comparison is fair.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fair evaluation of PPO agent vs baselines"
    )
    p.add_argument("--model",  type=str,   default=None)
    p.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--smoke",  action="store_true")
    p.add_argument("--full",   action="store_true",
                   help="Evaluate all saved PPO models + all baselines")
    p.add_argument("--sanity", action="store_true",
                   help="Run environment sanity check only")
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()

    if args.sanity:
        sanity_check(alpha=args.alpha, seed=args.seed)

    elif args.full:
        full_evaluation(smoke=args.smoke)

    elif args.model:
        df = evaluate_ppo(
            model_path=args.model,
            alpha=args.alpha,
            seed=args.seed,
            smoke=args.smoke,
            verbose=True,
        )
        out = os.path.join(RESULTS_DIR, "eval_single_run.csv")
        df.to_csv(out, index=False)
        print(f"\nEval CSV saved to {out}")

    else:
        print("Usage:")
        print("  Sanity check : python evaluate.py --sanity")
        print("  Single PPO   : python evaluate.py --model models/ppo_agent_alpha0.1_seed0_final.zip")
        print("  Full compare : python evaluate.py --full")
        print("  Smoke test   : python evaluate.py --full --smoke")
        