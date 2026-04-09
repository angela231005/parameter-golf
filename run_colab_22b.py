# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_22b Training Run
# **Fix of sota_22**: TTT AdamW LR corrected from 0.01 → 0.0003 (was 33× too large).
#
# **Root cause of sota_22 failure:**
# - `TTT_LR=0.01` with AdamW: the original note said "PR #1440: AdamW lr=0.01 beats SGD",
#   but lr=0.01 for AdamW is unstable — AdamW is already adaptive so a much lower LR is needed.
#   Correct value (verified in sota_28/29): `TTT_LR=0.0003`.
#
# **Everything else same as sota_22:**
# - RECUR_COUNT=2 (triple loop, PR #1450)
# - RECUR_LAYERS=2,3,4,5 (4 recurrent layers)
# - MTP_NUM_HEADS=2 (multi-token prediction)
# - Mousse optimizer (beta=0.95)
# - LAWA (k=15, freq=50)
# - GPTQ 128 AR seqs
# - Hash embedding 32768
#
# **Upgraded vs sota_22:**
# - Uses `train_gpt_sota_28.py` (latest code: recompile fixes, z-loss, skip_gates, Raki v6 WD)
# - Added Raki v6 WD: MUON_WD=0.090, EMBED_WD=0.090, ADAM_WD=0.02
# - SKIP_GATES_ENABLED=1
# - Z_LOSS_WEIGHT=0.0001
# - WARMDOWN_ITERS=4000 (was 6200: more steps at peak LR)

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
    # --- Architecture (from sota_22) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",           # was 6200: more steps at peak LR
    f"SWA_ENABLED=0",
    f"MTP_NUM_HEADS=2",
    f"MTP_LOSS_WEIGHT=0.1",
    f"VE_LAYERS=8,9,10",
    # --- Depth Recurrence (sota_22 config: 4 layers, triple loop) ---
    f"RECUR_LAYERS=2,3,4,5",
    f"RECUR_START_STEP=1500",
    f"RECUR_COUNT=2",                 # triple loop: each recur layer runs 2 extra times
    # --- Z-loss regularization ---
    f"Z_LOSS_WEIGHT=0.0001",
    # --- Raki v6 weight decay scheme ---
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
    # --- Legal Score-First TTT (BUG FIX: lr 0.01 → 0.0003) ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.0003",                 # FIX: was 0.01 (33× too large for AdamW)
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

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_28.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
