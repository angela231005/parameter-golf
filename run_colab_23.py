# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_23 Training Run
# Built on sota_22 with tuning from PR #1420 (Triple Loop + Fused Kernels, 5-seed 1.08309):
#
# **Changes vs sota_22:**
# - NO MTP (removed — empirically bad; also frees params for main model capacity)
# - RECUR_LAYERS=4,5 (not 2,3,4,5 — PR #1420 ablation: layers 4-5 clearly optimal;
#   "Loop layers 3-4: +0.00066 worse", "Loop layers 5-6: +0.00247 worse")
# - RECUR_START_STEP=2424 (0.35 × 6927 iterations — PR #1420 swept {0.30,0.35,0.40,0.50},
#   found 0.35 wins; our prior 1500=0.22 was too early — below 0.30 degrades quality)
# - PARALLEL_START_LAYER=7 (PR #1420 uses PARALLEL_RESIDUAL_START=7, not 5)
#
# **Inherited from sota_22 (unchanged):**
# - Mousse EMA optimizer (beta=0.95, PR #1440)
# - TTT AdamW lr=0.01 (PR #1440: beats SGD)
# - RECUR_COUNT=2 — triple loop: 3 total passes through recur layers
# - LAWA (k=15, freq=50)
# - GPTQ 11-candidate percentile grid + 128 AR calibration seqs
# - Hash embedding 32768 entries
# - max_fusion_size=16 (Triton SRAM OOM fix)
# - Warmdown 6200 iters

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
# ## 2. Install dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system("pip install -q sentencepiece zstandard brotli")
os.system('python3 -c "import sentencepiece, zstandard, brotli; print(\'deps OK\')"')

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

ITERATIONS = 6927

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
    # --- Architecture (from sota_10) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals: start at layer 7 (PR #1420: 7-10 is optimal layout) ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",        # Changed from 5 → 7 (PR #1420 ablation)
    # --- Training ---
    f"WARMDOWN_ITERS=6200",
    # NO MTP_NUM_HEADS / MTP_LOSS_WEIGHT — MTP empirically bad, omit entirely
    f"VE_LAYERS=8,9,10",
    f"RECUR_LAYERS=4,5",              # Changed from 2,3,4,5 → 4,5 (PR #1420: optimal)
    f"RECUR_START_STEP=2424",         # Changed from 1500 → 2424 (0.35×6927, PR #1420 optimal)
    f"RECUR_COUNT=2",                 # triple loop: 3 total passes (PR #1450/#1420)
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- Mousse optimizer (PR #1440) ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=128",
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.01",                   # AdamW uses higher LR (PR #1440)
    f"TTT_OPTIMIZER=adamw",           # AdamW TTT (PR #1440)
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_23.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
