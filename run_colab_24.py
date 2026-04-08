# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_24 Training Run
# Built on sota_23. Key change: **Raki v6 coprime-stride DistributedTokenLoader**
#
# **Changes vs sota_23:**
# - train_gpt_sota_24.py: replaced sequential TokenStream with Raki's coprime-stride multi-shard loader
#     - mmap-based zero-copy reads (no full shard load into RAM)
#     - coprime stride within each shard → full coverage without repetition bias
#     - adaptive shard weighting (alpha: 0.90→0.50 over training) → diversity increases over time
#
# **Inherited from sota_23 (unchanged):**
# - RECUR_LAYERS=4,5, RECUR_START_STEP=2424, RECUR_COUNT=1
# - Markov curriculum (RAKI_POWER=0.10)
# - Late QAT: last 200 steps + dynamo reset (LATE_QAT_STEPS=200)
# - Mousse EMA optimizer (beta=0.95)
# - TTT AdamW lr=0.0003 (10× smaller than Raki for safety)
# - LAWA (k=15, freq=50), GPTQ 128 AR seqs, Hash embed 32768
# - PARALLEL_START_LAYER=7, NO MTP, Warmdown 6200 iters

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
    # --- Architecture ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    # --- Training ---
    f"WARMDOWN_ITERS=6200",
    f"VE_LAYERS=8,9,10",
    f"RECUR_LAYERS=4,5",
    f"RECUR_START_STEP=2424",
    f"RECUR_COUNT=1",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- Mousse optimizer ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=128",
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.0003",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
    # --- Markov curriculum ---
    f"RAKI_POWER=0.10",
    # --- Late QAT ---
    f"LATE_QAT_STEPS=200",
    f"LATE_QAT_THRESHOLD=0",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_24.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
