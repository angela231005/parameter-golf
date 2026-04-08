# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_21 Training Run
# Built on sota_19 (sota_10 + TTT + N-gram + Hash) with training + quantization improvements:
#
# **Training improvements vs sota_20:**
# - MTP heads=2 (multi-token prediction aux loss during training, weight=0.1)
# - VE128 extended to layers 8,9,10 (3 layers)
# - Depth recurrence on layers 2,3,4,5 (4 layers)
# - Recurrence activated earlier: step 1500
# - LAWA enabled (lawa_k=15, lawa_freq=50) — averages last 15 checkpoints alongside EMA/SWA
# - Warmdown extended to 6200 steps (vs 5500 in sota_20)
#
# **Quantization improvements:**
# - GPTQ percentile clip grid: 5 → 11 candidates (finer search in [0.9975, 1.0])
# - GPTQ calibration sequences: 64 → 128 AR sequences (better Hessian estimate)
#
# **Eval improvements:**
# - TTT: 5 epochs (vs 3 in sota_20, more adaptation)
# - Eval-time hash embedding: 32768 entries (vs 16384, more bigram capacity)

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
    f"WARMDOWN_ITERS=6200",          # extended warmdown (sota_20: 5500)
    f"MTP_NUM_HEADS=2",              # multi-token prediction aux heads
    f"MTP_LOSS_WEIGHT=0.1",
    f"VE_LAYERS=8,9,10",             # 3 VE layers
    f"RECUR_LAYERS=2,3,4,5",         # 4 recur layers
    f"RECUR_START_STEP=1500",        # earlier recurrence activation
    # --- LAWA: average last 15 checkpoints (new in sota_21) ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- GPTQ calibration (128 AR seqs, 2x vs sota_20) ---
    f"GPTQ_AR_SEQS=128",             # better Hessian estimate (sota_20: 64)
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.002",
    f"TTT_EPOCHS=5",                 # more adaptation (sota_20: 3)
    f"TTT_CHUNK_SIZE=32768",
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding (larger table) ---
    f"HASH_EMB_SIZE=32768",          # more bigram capacity (sota_20: 16384)
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_21.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
