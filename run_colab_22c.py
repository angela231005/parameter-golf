# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_22c Training Run
# **Time-based training (4800s) + flash-attn + late QAT + TTT eval.**
#
# ## What changed vs sota_22b
#
# ### Change 1: Time-based training (MAX_WALLCLOCK_SECONDS=4800)
# 600s × 8 GPUs = 4800s total budget.
# ITERATIONS=999999 — effectively unlimited; training stops at wallclock cap instead.
# LR warmdown in train_gpt_sota_22c.py is already time-aware:
#   scale = remaining_ms / warmdown_ms  (not step-based when wallclock is set).
#
# ### Change 2: QAT later (LATE_QAT_THRESHOLD=0.10, QAT_START_STEP=0)
# New default in 22c: QAT fires at scale < 0.10 (last ~10% of warmdown).
# Old default 0.20 was too early; step-based trigger (QAT_START_STEP=4000) disabled.
# With time-based training, step counts are unpredictable → threshold-only is safer.
#
# ### Change 3: flash-attn native GQA
# train_gpt_sota_22c.py installs flash-attn at startup and falls back to SDPA.
# Removes K/V repeat_interleave copy (4→8 heads) on every attention call.
#
# ### Change 4: TTT_CHUNK_SIZE=65536 + GPTQ_AR_SEQS=128 (runtime)
# Halved Python loop iterations during TTT eval, same total FLOPs.
# Halved GPTQ Hessian calibration passes.
#
# ### Change 5: Best TTT settings from sota_34/35
# TTT_LR=0.005, TTT_OPTIMIZER=adamw, TTT_EPOCHS=5, TTT_FREEZE_BLOCKS=0
# These are the settings verified in PR #1413 / PR #1437.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Clone repo

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import glob
import os

REPO_URL = "https://github.com/angela231005/parameter-golf"
REPO_DIR = "parameter-golf"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Dependencies are auto-installed by train_gpt_sota_22c.py
# (sentencepiece, zstandard, brotli, flash-attn — no manual step needed)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42          # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    # Time-based training: run until wallclock cap, not a fixed step count.
    f"ITERATIONS=999999",
    f"MAX_WALLCLOCK_SECONDS=4800",  # 600s × 8 GPUs
    f"TARGET_MB={TARGET_MB}",
    # --- Architecture (from sota_22) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Training improvements ---
    f"WARMDOWN_ITERS=6200",
    f"MTP_NUM_HEADS=0",
    f"MTP_LOSS_WEIGHT=0.1",
    f"VE_LAYERS=8,9,10",
    # --- Depth Recurrence ---
    f"RECUR_LAYERS=2,3,4,5",
    f"RECUR_START_STEP=1500",
    f"RECUR_COUNT=2",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- Mousse optimizer ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=128",  # halved vs 256, still 4× default
    # --- QAT: late, threshold-only (no hard step trigger) ---
    f"LATE_QAT_THRESHOLD=0.10",  # fires at last ~10% of warmdown (was 0.20)
    f"QAT_START_STEP=0",          # disabled: time-based training = unpredictable step count
    # --- Score-First TTT (best settings from sota_34/35) ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.005",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_SIZE=65536",  # 50% fewer loop iterations, same FLOPs
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram Tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_22c.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
