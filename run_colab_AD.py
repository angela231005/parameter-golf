# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — Strategy AD Training Run
# **Branch `ml` · train_gpt_strategy_AD.py**
#
# ## What this script does (5 modifications over baseline)
#
# ### Mod 1 — MuonEq-R (row-norm before Newton-Schulz5)
# Row-normalise each gradient shard **before** NS5 inside Muon.
# Forces unit-row-norm spectrum → more uniform update magnitudes.
#
# ### Mod 2 — Depth Recurrence
# Selected layers (RECUR_LAYERS) are replayed RECUR_COUNT extra times per
# forward pass after RECUR_START_STEP steps, reusing weights for free.
#
# ### Mod 3 — Tuned Weight Decay
# MUON_WD=0.095 and EMBED_WD=0.095 (vs 0.02 default) for better final BPB.
#
# ### Mod 4 — Pre-quant TTT + optional eval-time TTT
# **Pre-quant TTT** (PREQUANT_TTT_ENABLED=1): lightweight AdamW fine-tune on
# calibration data BEFORE GPTQ — result is baked into the quantized artifact.
# **Eval-time TTT** (TTT_ENABLED): optional second TTT pass on the int6 model.
#
# ### Mod 5 — Parallel Residuals
# Attention and MLP run in parallel from PARALLEL_START_LAYER onward.
#
# ### Bonus — EMA + XSA
# EMA (EMA_DECAY) shadow model for eval stability.
# XSA_LAST_N enables cross-shard attention on the last N layers.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Clone repo (branch: ml)

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import glob
import os

REPO_URL = "https://github.com/angela231005/parameter-golf"
REPO_DIR = "parameter-golf"
BRANCH = "ml"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} fetch origin")
    os.system(f"git -C {REPO_DIR} checkout {BRANCH}")
    os.system(f"git -C {REPO_DIR} pull --ff-only")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Install dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system("pip install -q sentencepiece zstandard brotli")
os.system('python3 -c "import sentencepiece, zstandard, brotli; print(\'deps OK\')"')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42           # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
# DATA_DIR must contain:
#   datasets/fineweb10B_sp{VOCAB_SIZE}/fineweb_{train,val}_*.bin
#   tokenizers/fineweb_{VOCAB_SIZE}_bpe.model
DATA_DIR = "/kaggle/input/datasets/haphmph/parameter-golf/data"
VOCAB_SIZE = 1024

env = " ".join([
    f"SEED={SEED}",
    f"DATA_DIR={DATA_DIR}",
    f"VOCAB_SIZE={VOCAB_SIZE}",
    f"TARGET_MB={TARGET_MB}",

    # --- Training schedule ---
    f"ITERATIONS=999999",           # stopped by wallclock cap
    f"MAX_WALLCLOCK_SECONDS=4800",  # 600s x 8 GPUs
    f"WARMDOWN_FRAC=0.667",
    f"WARMUP_STEPS=20",
    f"VAL_LOSS_EVERY=4000",
    f"TRAIN_LOG_EVERY=500",

    # --- Architecture ---
    f"NUM_LAYERS=11",
    f"MODEL_DIM=512",
    f"NUM_HEADS=8",
    f"NUM_KV_HEADS=4",
    f"MLP_MULT=4.0",
    f"QK_GAIN_INIT=5.0",
    f"LOGIT_SOFTCAP=30.0",
    f"ROPE_BASE=10000.0",
    f"ROPE_DIMS=16",
    f"SKIP_GATES_ENABLED=1",
    f"TIE_EMBEDDINGS=1",
    f"VE_ENABLED=1",
    f"VE_DIM=128",
    f"VE_LAYERS=9,10",

    # --- XSA: cross-shard attention on last N layers ---
    f"XSA_LAST_N=11",               # 0 to disable

    # --- Mod 2: Depth Recurrence ---
    f"RECUR_LAYERS=3,4,5",
    f"RECUR_START_STEP=3000",

    # --- Mod 5: Parallel Residuals ---
    f"PARALLEL_START_LAYER=7",

    # --- Optimizer (Mod 1: MuonEq-R; Mod 3: tuned WD) ---
    f"MATRIX_LR=0.022",
    f"SCALAR_LR=0.02",
    f"EMBED_LR=0.6",
    f"HEAD_LR=0.008",
    f"TIED_EMBED_LR=0.03",
    f"MUON_MOMENTUM=0.99",
    f"MUON_BACKEND_STEPS=5",
    f"MUON_WD=0.095",               # Mod 3 (was 0.02)
    f"ADAM_WD=0.02",
    f"EMBED_WD=0.095",              # Mod 3
    f"BETA1=0.9",
    f"BETA2=0.95",
    f"GRAD_CLIP_NORM=0.3",
    f"EMA_DECAY=0.9965",

    # --- Mod 4a: Pre-quant TTT --- DISABLED (uses val data before eval = rules violation)
    f"PREQUANT_TTT_ENABLED=0",
    f"PREQUANT_TTT_LR=0.0004",
    f"PREQUANT_TTT_EPOCHS=15",
    f"PREQUANT_TTT_FREEZE_BLOCKS=0",
    f"PREQUANT_TTT_BATCH_SEQS=32",
    f"PREQUANT_TTT_COSINE_DECAY=1",

    # --- Mod 4b: Eval-time TTT on final int6 (optional) ---
    f"TTT_ENABLED=0",               # set to 1 to enable
    f"TTT_LR=0.005",
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_TOKENS=65536",
    f"TTT_FREEZE_BLOCKS=0",

    # --- GPTQ compression ---
    f"GPTQ_ENABLED=1",
    f"GPTQ_CALIBRATION_BATCHES=64",
    f"COMPRESSOR=brotli",
    f"SDCLIP_K=12.85",

    # --- Eval ---
    f"SLIDING_WINDOW_ENABLED=1",
    f"EVAL_STRIDE=32",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_strategy_AD.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)