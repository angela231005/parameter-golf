# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_AD2 Training Run
# **Developed from strategy_AD — branch `main`**
#
# ## What AD already had (unchanged here)
# - MuonEq-R: row-normalise gradient before NS5 inside Muon
# - Skip gates: learnable sigmoid gates on U-Net skip connections
# - Parallel residuals: attention lane + MLP lane merged with learned α
# - Depth recurrence: repeat recur_layers at recur_start_step
# - EMA weight averaging (now with dynamic decay schedule)
# - Score-first TTT (now with AdamW)
#
# ## New in AD2 (NOT from sota_22c)
#
# ### [AD2-1] RECUR_COUNT — Multi-step depth recurrence
# Repeat recur_layers N times instead of just once.
# Default RECUR_COUNT=2 → 3 total passes through the same weights.
# Free compute (weights shared); only activation memory grows.
# Virtual layout: [0..R-1, recur_layers×RECUR_COUNT, R..N-1]
#
# ### [AD2-2] TTT AdamW
# Replace SGD+momentum with AdamW in test-time training.
# AdamW adapts per-parameter learning rate → faster convergence per chunk.
#
# ### [AD2-3] Dynamic EMA decay
# - frac < 0.3 → ema_decay_early (faster, captures early exploration)
# - frac 0.3–0.7 → linear blend toward ema_decay_late
# - frac > 0.7 → ema_decay_late (slower, stabilise near optimum)
# Net effect: EMA tracks model well early, then freezes near the best weights late.

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
# ## 2. Dependencies are auto-installed by train_gpt_sota_AD2.py

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42           # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
DATA_DIR = "/kaggle/input/datasets/haphmph/parameter-golf/data"

env = " ".join([
    f"SEED={SEED}",
    f"DATA_DIR={DATA_DIR}",
    f"TARGET_MB={TARGET_MB}",

    # --- Training schedule ---
    f"ITERATIONS=999999",           # stopped by wallclock cap
    f"MAX_WALLCLOCK_SECONDS=4800",  # 600s x 8 GPUs
    f"WARMDOWN_FRAC=0.667",
    f"WARMUP_STEPS=20",
    f"VAL_LOSS_EVERY=4000",
    f"TRAIN_LOG_EVERY=500",

    # --- Architecture (from AD) ---
    f"NUM_LAYERS=11",
    f"MODEL_DIM=512",
    f"NUM_HEADS=8",
    f"NUM_KV_HEADS=4",
    f"QK_GAIN_INIT=5.0",
    f"SKIP_GATES_ENABLED=1",
    f"PARALLEL_START_LAYER=7",
    f"VE_LAYERS=9,10",

    # --- Optimizer (MuonEq-R built in, no Mousse) ---
    f"MATRIX_LR=0.022",
    f"MUON_WD=0.095",
    f"EMBED_WD=0.095",
    f"MUON_MOMENTUM=0.99",
    f"MUON_MOMENTUM_WARMUP_STEPS=1500",

    # --- [AD2-1] Multi-step Depth Recurrence ---
    # RECUR_COUNT=2 → 3 total passes through recur_layers
    # Earlier activation than AD (2000 vs 3000) since model is more capable now
    f"RECUR_LAYERS=3,4,5",
    f"RECUR_START_STEP=2000",
    f"RECUR_COUNT=2",

    # --- [AD2-3] Dynamic EMA ---
    f"EMA_DECAY=0.9965",
    f"EMA_DECAY_EARLY=0.990",
    f"EMA_DECAY_LATE=0.9990",

    # --- [AD2-2] TTT with AdamW ---
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_BETA1=0.9",
    f"TTT_BETA2=0.99",
    f"TTT_LR=0.005",
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_TOKENS=32768",
    f"TTT_FREEZE_BLOCKS=0",

    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=128",

    # --- Pre-quant TTT --- DISABLED (uses val data before eval = rules violation)
    f"PREQUANT_TTT_ENABLED=0",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_AD2.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
