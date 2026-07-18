"""
config.py — Central configuration for Threat-Adaptive Federated Learning system.
All hyperparameters are defined here. Never hardcode values in other modules.
"""

import os

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
GLOBAL_SEED = 42

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
DATA_DIR      = os.path.join(BASE_DIR, "data")

for _d in [RESULTS_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DATASET           = "CIFAR10"          # "CIFAR10" | "FEMNIST"
NUM_CLASSES       = 10
NUM_CHANNELS      = 3
IMAGE_SIZE        = 32

# ─────────────────────────────────────────────
# FEDERATION
# ─────────────────────────────────────────────
NUM_CLIENTS            = 20
CLIENTS_PER_ROUND      = 10           # 50 % participation
MIN_SAMPLES_PER_CLIENT = 200
LOCAL_EPOCHS           = 5
LOCAL_BATCH_SIZE       = 32
LOCAL_LR               = 0.01
LOCAL_MOMENTUM         = 0.9

# Dirichlet concentration values to sweep
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0]
DEFAULT_ALPHA = 0.1                   # used in smoke-test / single runs

# ─────────────────────────────────────────────
# TRAINING SCHEDULE
# ─────────────────────────────────────────────
FULL_ROUNDS = 200       # rounds per episode (keep this)
PPO_TOTAL_EPISODES = 10 # train across 10 full episodes
SMOKE_ROUNDS  = 20                    # fast test

# Server-side clean root dataset for FLTrust
FLTRUST_ROOT_SIZE = 100

# ─────────────────────────────────────────────
# THREAT INJECTION
# ─────────────────────────────────────────────
GIA_CLIENTS       = 3                 # passive adversaries recording gradients
BACKDOOR_CLIENTS  = 2
BYZANTINE_CLIENTS = 2

BACKDOOR_TRIGGER_SIZE  = 3            # 3×3 white patch
BACKDOOR_POISON_RATE   = 0.15         # 15 % of local data poisoned
BACKDOOR_TARGET_CLASS  = 0

BYZANTINE_SCALE   = -5.0             # sign-flip + amplification

# GIA computation every N rounds (interpolated between)
GIA_EVAL_INTERVAL = 5
GIA_OPT_STEPS     = 100              # DLG inner optimisation steps (reduced for speed)
GIA_LR            = 0.1

# GIA reconstruction batch size — must match a SINGLE forward/backward step,
# not the multi-epoch local_train() delta. See client.py::compute_single_step_gradient.
GIA_BATCH_SIZE     = 4

# ─────────────────────────────────────────────
# STATE SPACE  (13-dimensional)
# ─────────────────────────────────────────────
STATE_DIM       = 15
NUM_DEFENSES    = 5                   # actions 0-4

# Running normaliser momentum (1 - this = update weight)
NORM_MOMENTUM   = 0.99

# ─────────────────────────────────────────────
# DEFENSE PARAMETERS
# ─────────────────────────────────────────────
# Action 0 — DP-SGD
DP_NOISE_MULTIPLIER = 1.1
DP_CLIP_NORM        = 1.0

# Action 1 — FLTrust  (root dataset size defined above)

# Action 2 — Multi-Krum
KRUM_F = 2                            # tolerate f Byzantine clients

# Action 3 — FLAME
FLAME_MIN_CLUSTER = 3                 # kept for reference / fallback path only
                                       # (HDBSCAN is no longer the clustering
                                       # method used — see defenses.py)

# Action 4 — NoDefense  (plain FedAvg)

# Cost penalties (fixed, not measured at runtime)
DEFENSE_COST = {
    0: 0.3,   # DP-SGD
    1: 0.2,   # FLTrust
    2: 0.1,   # Krum
    3: 0.5,   # Flame
    4: 0.0,   # NoDefense
}

# ─────────────────────────────────────────────
# REWARD FUNCTION WEIGHTS
# ─────────────────────────────────────────────
# FIX (see project changelog): these were declared here but state_reward.py's
# compute_reward() hardcoded its own 0.25/0.25/0.30/0.10/0.10 split instead
# of reading them — sweeping these had zero effect. Now wired through
# properly. Values below are set to MATCH what was actually hardcoded (and
# already empirically validated across your seed0/seed1/alpha0.3 runs), so
# fixing the wiring does not silently change training behavior right before
# the deadline. R_ACC is split 50/50 internally between delta_acc and
# acc_level (see state_reward.compute_reward). Change these deliberately,
# as a documented ablation, not as a side effect of this fix.
R_ACC       = 0.50   # split 0.25 delta_acc + 0.25 acc_level
R_PRIVACY   = 0.30
R_COST      = 0.10
R_UTILITY   = 0.10

REWARD_NORM_MOMENTUM = 0.99           # running reward normaliser

# ─────────────────────────────────────────────
# PPO HYPERPARAMETERS
# ─────────────────────────────────────────────
PPO_LR_START    = 3e-4
PPO_LR_END      = 1e-5
PPO_CLIP_EPS    = 0.2
PPO_ENT_COEF    = 0.05
PPO_VF_COEF     = 0.5
PPO_GAE_LAMBDA  = 0.95
PPO_GAMMA       = 0.99
PPO_BATCH_SIZE  = 40
PPO_N_EPOCHS    = 10
PPO_N_STEPS     = 200               # rollout buffer length
PPO_N_ENVS      = 5                   # parallel environments

POLICY_NET_ARCH = [128, 64]           # shared layers; SB3 adds heads automatically

# ─────────────────────────────────────────────
# TRAINING RUNS
# ─────────────────────────────────────────────
NUM_SEEDS         = 5
CHECKPOINT_EVERY  = 50                # rounds

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_ROUNDS            = 50
EVAL_THREAT_CONFIG     = "all"        # all three threats active simultaneously
COMPOSITE_W_ACC        = 0.4
COMPOSITE_W_PRIVACY    = 0.3
COMPOSITE_W_ASR        = 0.3

# ─────────────────────────────────────────────
# DEBUGGING
# ─────────────────────────────────────────────
# Set True to print a per-stage timing breakdown for the first 5 rounds of
# the next episode, then auto-disable. Use this to diagnose real-hardware
# bottlenecks (e.g. GPU transfer overhead) — see fl_env.py.
DEBUG_TIMING = False

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")