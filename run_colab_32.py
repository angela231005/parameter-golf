# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_32 Training Run
# **EMA_DECAY=0.9965 + RECUR_LAYERS=4,5 on clean WD=0.04 baseline.**
#
# ## Context
# PR #1435 (1.0980 BPB, beats 1.1147 by −0.0167) uses two key ideas:
# 1. **EMA decay 0.9965** instead of standard 0.997 — more aggressive recent-checkpoint
#    weighting during the EMA average applied at end of training.
# 2. **RECUR_LAYERS=4,5 at step 3000** — 13 virtual layers from 11 physical, +0 artifact cost.
#
# Both are clean training-time changes, no eval-time trickery.
#
# ## SLOT reasoning
# SLOT (PR #1313, 0.8637 BPB) optimizes delta on the exact tokens it then scores — a
# 2-pass retroactive approach that is NOT causal. PR #1313 Copilot review flagged this.
# All SLOT PRs remain unmerged. Not included here.
#
# ## Stack vs sota_31
# | Change                   | sota_31  | sota_32  |
# |--------------------------|----------|----------|
# | MUON_WD / EMBED_WD       | 0.04     | 0.04     |
# | ADAM_WD                  | 0.04     | 0.04     |
# | EMA_DECAY                | 0.997    | **0.9965** (PR #1435) |
# | RECUR_LAYERS             | 3,4,5    | **4,5** at step 3000 (PR #1435) |
# | RECUR_START_STEP         | 3000     | 3000     |
# | TTT_ENABLED              | 0        | 0        |
# | SLOT_ENABLED             | 0        | 0        |
# | Mousse                   | 1        | 1        |
# | BIGRAM_VOCAB_SIZE         | 3072     | 3072     |
#
# ## Expected outcome
# ~1.095–1.10 BPB (WD=0.04 baseline + EMA=0.9965 + cleaner RECUR config)

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
    # --- Architecture (exact record values) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_VOCAB_SIZE=3072",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    f"SWA_ENABLED=1",
    f"SWA_EVERY=50",
    f"VE_LAYERS=9,10",
    # --- WD matching the 1.1147 record (PRIMARY FIX from sota_30/31) ---
    f"MUON_WD=0.04",
    f"EMBED_WD=0.04",
    f"ADAM_WD=0.04",
    # --- EMA decay: PR #1435 shows 0.9965 > 0.997 (+0.017 BPB) ---
    f"EMA_DECAY=0.9965",
    # --- RECUR: PR #1435 uses layers 4,5 from step 3000 (13 virtual from 11 physical) ---
    f"RECUR_LAYERS=4,5",
    f"RECUR_START_STEP=3000",
    # --- Mousse optimizer (sota_31 baseline) ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=256",
    # --- TTT disabled ---
    f"TTT_ENABLED=0",
    # --- SLOT disabled (retroactive 2-pass scoring — not causal) ---
    f"SLOT_ENABLED=0",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
    # --- Markov curriculum ---
    f"RAKI_POWER=0.10",
    # --- Late QAT ---
    f"LATE_QAT_STEPS=200",
    f"LATE_QAT_THRESHOLD=0",
    # --- NS steps matching record ---
    f"MUON_BACKEND_STEPS=5",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_28.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
