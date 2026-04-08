# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_22 Training Run
# Built on sota_21 with insights from PR #1440 (Raki v6) + PR #1450 (TMA Triple Loop):
#
# **New in sota_22:**
# - Mousse optimizer: diagonal Kronecker curvature preconditioning before Newton-Schulz
#   (arXiv:2603.09697, PR #1440 reports -0.002 BPB; applied with EMA β=0.95)
# - TTT AdamW: use AdamW instead of SGD for test-time training
#   (PR #1440: AdamW lr=0.01 beats SGD lr=0.002 for depth-recurrent models)
# - RECUR_COUNT=2: triple loop (each recur layer repeated 2 extra times = 3 total passes)
#   (PR #1450: NUM_LOOPS=3 → 17 virtual layers from 11 physical → 1.08480 BPB)
#
# **Inherited from sota_21:**
# - GPTQ percentile clip grid: 11 candidates (finer search)
# - GPTQ 128 AR calibration sequences
# - LAWA enabled (k=15, freq=50)
# - Hash embedding: 32768 entries
# - max_fusion_size=16 (Triton SRAM OOM fix)

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
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Training improvements ---
    f"WARMDOWN_ITERS=6200",
    f"MTP_NUM_HEADS=2",
    f"MTP_LOSS_WEIGHT=0.1",
    f"VE_LAYERS=8,9,10",
    f"RECUR_LAYERS=2,3,4,5",
    f"RECUR_START_STEP=1500",
    f"RECUR_COUNT=2",             # triple loop: each recur layer runs 2 extra times (PR #1450)
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- Mousse optimizer (new in sota_22, PR #1440) ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- GPTQ calibration (128 AR seqs) ---
    f"GPTQ_AR_SEQS=128",
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.01",               # AdamW uses higher LR than SGD (PR #1440)
    f"TTT_OPTIMIZER=adamw",       # AdamW TTT (PR #1440: beats SGD)
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_22.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
