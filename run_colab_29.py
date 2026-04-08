# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_29 Training Run
# Built on sota_28 code. **Focus: strip complexity, match the 1.1147 record config more closely.**
#
# ## Root-cause analysis of sota_22–28 failures
#
# | Version | Failure mode |
# |---|---|
# | sota_26 | skip_gates neutral; high MUON_WD=0.090 (record=0.04) may over-regularise |
# | sota_27 | LAWA replaces SWA but the 1.1147 record uses tight SWA(every 50) + EMA |
# | sota_28 | Crashed (recompile_limit) + seq-length curriculum hurts optimization |
#
# ## Changes vs sota_28 (run config only — same train_gpt_sota_28.py code)
#
# ### Removed (proven harmful or unproven):
# - **TTT disabled** (`TTT_ENABLED=0`): the 1.1147 record tried TTT ≥25 times — "neutral or
#   negative on this stack." Full Hessian GPTQ already learns optimal quantized representations;
#   TTT on top is redundant and wastes eval time from the 10-min budget.
# - **Seq-length curriculum removed**: steps 0→1500 at seq_len=512 expose the model to
#   shorter contexts only, starving it of long-range patterns during early high-LR exploration.
#   Constant seq_len=2048 throughout is safer and matches all competing configs.
# - **RECUR reverted to 4,5 with later start** (`RECUR_LAYERS=4,5`, `RECUR_START_STEP=3000`):
#   starting recurrence at step 2000 inside the high-LR phase disrupted the loss trajectory.
#   Starting at step 3000 (inside warmdown) is more stable.
#
# ### Increased:
# - **GPTQ_AR_SEQS=256** (was 128): the 1.1147 record uses 256 calibration sequences for Hessian
#   estimation. Doubling gives a better H = X^T X estimate → less quantization error.
#
# ### Unchanged from sota_28:
# - Z-loss (`Z_LOSS_WEIGHT=1e-4`): the record uses this too
# - Sigmoid skip_gates (SKIP_GATES_ENABLED=1)
# - Raki v6 WD: MUON_WD=0.090, EMBED_WD=0.090, ADAM_WD=0.02
# - LAWA (k=15, freq=50)
# - Mousse EMA optimizer (beta=0.95)
# - WARMDOWN_ITERS=4000
# - BigramHash 3072×112, XSA all 11 layers, Partial RoPE (dims=16)
# - Full Hessian GPTQ with AR self-gen calibration
# - Selective pruning, LZMA preset=9

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
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    f"SWA_ENABLED=0",                 # LAWA is our weight averaging
    f"VE_LAYERS=8,9,10",
    # --- Depth Recurrence (conservative: same 2 layers, later start) ---
    f"RECUR_LAYERS=4,5",              # was 3,4,5: drop the extra layer that disrupts early training
    f"RECUR_START_STEP=3000",         # was 2000: start inside warmdown (safer)
    f"RECUR_COUNT=1",
    # --- Z-loss regularization (record uses 1e-4 too) ---
    f"Z_LOSS_WEIGHT=0.0001",
    # --- NO seq-length curriculum ---
    # (removed: shorter seqs early starve long-range learning during high-LR phase)
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
    # --- GPTQ calibration (256 seqs = record-level Hessian quality) ---
    f"GPTQ_AR_SEQS=256",
    # --- TTT DISABLED (record: 25 attempts, neutral/negative on Full-GPTQ stack) ---
    f"TTT_ENABLED=0",
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding (still used even without TTT) ---
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
