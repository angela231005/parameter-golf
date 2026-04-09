# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_31 Training Run
# **Hypothesis: Mousse optimizer helps when WD is correctly set.**
#
# ## Context
# sota_30 fixes MUON_WD=0.04 (the root cause of 25–29 failures) and also changes
# BIGRAM 3072×112 → 4096×96. sota_31 isolates the true signal from sota_30 by:
# 1. Reverting BIGRAM to exact record values (3072×112) — removes a confound
# 2. Adding **Mousse optimizer** as the single new variable
#
# ## Why Mousse now?
# Mousse (diagonal Kronecker curvature preconditioning, arXiv:2603.09697) was present in
# sota_25–29 but those runs had MUON_WD=0.090 poisoning everything. With MUON_WD=0.04
# (matching the proven record), Mousse gets its first clean test.
# Raki v6 reports −0.002 BPB from Mousse on a similar stack. Zero artifact cost.
#
# ## Our free advantages vs the 1.1147 record (no env var needed)
# - **Trigram hashing**: our train_gpt_sota_28.py hardcodes TRIGRAM="1" in GPT.__init__,
#   the record has TRIGRAM="0". Free (t-2, t-1, t) pattern lookups into same embed table.
# - **Coprime-stride DistributedTokenLoader**: better data diversity across batches.
# - **Markov curriculum** (RAKI_POWER=0.10): n-gram biased sampling during training.
# - **Parallel residual from layer 7** (record has no PARALLEL_START_LAYER).
#
# ## Changes vs sota_30
# - BIGRAM_VOCAB_SIZE=3072 (was 4096: revert to exact record value, remove confound)
# - BIGRAM_DIM=112 (was 96: revert to exact record value)
# - MOUSSE_ENABLED=1 (was 0: single novel addition, first clean test with WD=0.04)
# - MOUSSE_BETA=0.95

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
    f"BIGRAM_VOCAB_SIZE=3072",         # match record exactly (sota_30 tried 4096, reverted)
    f"BIGRAM_DIM=112",                 # match record exactly (sota_30 tried 96, reverted)
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    f"SWA_ENABLED=1",
    f"SWA_EVERY=50",
    f"VE_LAYERS=9,10",
    # --- WD matching the 1.1147 record (PRIMARY FIX from sota_30) ---
    f"MUON_WD=0.04",
    f"EMBED_WD=0.04",
    f"ADAM_WD=0.04",
    # --- Novel: Mousse optimizer — first clean test with correct WD ---
    f"MOUSSE_ENABLED=1",               # was 0 in sota_30; Raki v6 claims -0.002 BPB
    f"MOUSSE_BETA=0.95",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=256",
    # --- TTT disabled ---
    f"TTT_ENABLED=0",
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
