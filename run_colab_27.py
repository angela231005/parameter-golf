# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_27 Training Run
# Built on sota_26. Key change: **WARMDOWN_ITERS 6200→4000 + SWA_ENABLED=0**
#
# **Changes vs sota_26:**
# - WARMDOWN_ITERS=4000 (was 6200)
#     - Peak-LR phase: ~2927 steps (42% of training) instead of 727 steps (10%)
#     - Matches the 1.1147 BPB record (2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072)
#     - More exploration at high LR → better minima before warmdown compression
# - SWA_ENABLED=0 (was 1)
#     - Raki v6 has swa_enabled=False
#     - We already have LAWA (k=15, freq=50) which is a better rolling average
#     - SWA (scale < 0.2 trigger) and LAWA are redundant; LAWA wins
#
# **Inherited from sota_26 (unchanged, uses train_gpt_sota_26.py):**
# - Sigmoid skip_gates on U-Net skip connections (SKIP_GATES_ENABLED=1)
# - Raki v6 weight decay: MUON_WD=0.090, EMBED_WD=0.090, ADAM_WD=0.02
# - Raki coprime-stride DistributedTokenLoader
# - Markov curriculum (RAKI_POWER=0.10)
# - Late QAT: last 200 steps + dynamo reset
# - Mousse EMA optimizer (beta=0.95)
# - TTT AdamW lr=0.0003
# - RECUR_LAYERS=4,5, RECUR_START_STEP=2424, RECUR_COUNT=1
# - LAWA (k=15, freq=50), GPTQ 128 AR seqs, Hash embed 32768
# - PARALLEL_START_LAYER=7, NO MTP

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
    f"WARMDOWN_ITERS=4000",           # was 6200; record uses 4000 → 2927 steps at peak LR (42%)
    f"SWA_ENABLED=0",                 # was 1; Raki v6 off; LAWA (k=15, freq=50) is better
    f"VE_LAYERS=8,9,10",
    f"RECUR_LAYERS=4,5",
    f"RECUR_START_STEP=2424",
    f"RECUR_COUNT=1",
    # --- Raki v6 weight decay scheme (Modification 3) ---
    f"MUON_WD=0.090",
    f"EMBED_WD=0.090",
    f"ADAM_WD=0.02",
    # --- Raki v6 sigmoid skip gates ---
    f"SKIP_GATES_ENABLED=1",
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

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_26.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
