# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_22AD Training Run
# **sota_22c + strategy_AD \u2014 branch `main`**
#
# ## What changed vs sota_22c
#
# ### Mod A — MuonEq-R (from strategy_AD)
# Row-normalise each gradient shard **before** Newton-Schulz5 inside Muon.
# Forces unit-row-norm spectrum \u2192 more uniform update magnitudes.
#
# ### Mod D — Pre-quant AdamW TTT (from strategy_AD / PR #1423)
# After training finishes (EMA/SWA/LAWA weight selection), run a lightweight
# AdamW fine-tune on validation data **before GPTQ**.  Result is baked into the
# quantized artifact.  Controlled by PREQUANT_TTT_ENABLED / PREQUANT_TTT_*.
#
# ### Plus — EMA_DECAY now configurable (was hardcoded 0.997)
#
# ## Everything else is unchanged from sota_22c
# (bigram, LAWA, Mousse, MTP, n-gram tilt, hash-emb, QAT, time-based training,
#  parallel residuals, depth recurrence, XSA, VE, sliding-window eval, GPTQ)

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
# ## 2. Dependencies are auto-installed by train_gpt_sota_22AD.py

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42           # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"TARGET_MB={TARGET_MB}",

    # --- Training schedule ---
    f"ITERATIONS=999999",           # stopped by wallclock cap
    f"MAX_WALLCLOCK_SECONDS=4800",  # 600s x 8 GPUs
    f"WARMDOWN_ITERS=6200",
    f"WARMUP_STEPS=20",
    f"VAL_LOSS_EVERY=4000",
    f"TRAIN_LOG_EVERY=500",

    # --- Architecture (from sota_22) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",

    # --- Training improvements ---
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
    f"GPTQ_AR_SEQS=128",

    # --- QAT: late, threshold-only ---
    f"LATE_QAT_THRESHOLD=0.10",
    f"QAT_START_STEP=0",

    # --- Mod A: MuonEq-R (built into optimizer, no flag needed) ---

    # --- Mod D: Pre-quant AdamW TTT --- DISABLED (uses val data before eval = rules violation)
    f"PREQUANT_TTT_ENABLED=0",
    f"PREQUANT_TTT_LR=0.0004",
    f"PREQUANT_TTT_EPOCHS=15",
    f"PREQUANT_TTT_FREEZE_BLOCKS=0",
    f"PREQUANT_TTT_BATCH_SEQS=32",
    f"PREQUANT_TTT_COSINE_DECAY=1",
    f"EMA_DECAY=0.997",

    # --- Score-First TTT (best settings from sota_34/35) ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.005",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_SIZE=65536",
    f"TTT_FREEZE_BLOCKS=0",

    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",

    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_22AD.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)