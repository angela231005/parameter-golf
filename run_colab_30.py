# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_30 Training Run
# **Core hypothesis: MUON_WD=0.090 is the root cause of all sota_25–29 failures.**
#
# ## Root cause analysis
#
# Every run since sota_25 used `MUON_WD=0.090` (from the Raki v6 recommendation).
# The 1.1147 record — the ONLY entry that actually beat the previous SOTA — uses `MUON_WD=0.04`.
# That is **2.25× more weight decay** applied to every attention and MLP weight matrix.
# Higher WD → smaller weight magnitudes → weaker representations → systematic underfit.
# This single parameter explains why 25, 26, 27, 28, 29 all underperformed despite adding
# individually interesting features on top of a poisoned baseline.
#
# ## Changes vs sota_29
#
# ### Primary fix (high confidence):
# - **MUON_WD=0.04** (was 0.090) — match the proven winning config
# - **EMBED_WD=0.04** (was 0.090) — consistent
# - **ADAM_WD=0.04** (was 0.02) — consistent with record
# - **SWA_ENABLED=1, SWA_EVERY=50** — re-enable tight SWA (record uses EMA + SWA; LAWA stays on too)
#
# ### Novel addition (bigram vocab trade-off):
# - **BIGRAM_VOCAB_SIZE=4096** (was 3072) — 33% more distinct bigram buckets
# - **BIGRAM_DIM=96** (was 112) — slightly smaller per-bucket projection
#   Net artifact delta: +91KB (safe: 15.87MB + 0.09MB = ~15.96MB)
#   Math: embed table 4096×96×2B=786KB vs 3072×112×2B=688KB (+98KB);
#         GPTQ projection 96→512 saves ~7KB vs 112→512. Net: +91KB.
#   Rationale: 4096 buckets covers more unique (t-1, t) bigram pairs before hash collision.
#   With 1024 vocab, there are 1024²=1M possible bigrams; 4096 buckets → 256× hash table.
#
# ### Stripped (unproven on this stack):
# - MOUSSE_ENABLED=0 (record doesn't use it; added noise to sota_25–29)
# - SKIP_GATES_ENABLED=0 (not in record; neutral or harmful with high WD)
# - Z_LOSS_WEIGHT=0 (1.1147 record doesn't use Z-loss)
# - RECUR_START_STEP=3000 → **no RECUR at all** (record has no depth recurrence)
#
# ### Unchanged from sota_29:
# - TTT_ENABLED=0 (record-proven: 25 failed attempts on Full-GPTQ stack)
# - GPTQ_AR_SEQS=256 (high-quality Hessians)
# - Full Hessian GPTQ + AR self-gen calibration
# - LAWA k=15, freq=50 (our addition to record baseline; keep it)
# - VE_LAYERS=9,10 (match record exactly; was 8,9,10)
# - WARMDOWN_ITERS=4000
# - BigramHash trigram ON (default in code), XSA all 11 layers
# - Partial RoPE (dims=16), MUON_BACKEND_STEPS=5 (match record)

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
    # --- Novel: larger bigram vocab, slightly smaller dim ---
    # +33% more bigram buckets (4096 vs 3072), net +91KB artifact
    f"BIGRAM_VOCAB_SIZE=4096",         # was 3072
    f"BIGRAM_DIM=96",                  # was 112 (saves 7KB on GPTQ projection)
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    # --- SWA: re-enable tight SWA matching record (EMA always on in code) ---
    f"SWA_ENABLED=1",                  # was 0; record uses EMA(0.997) + SWA(50)
    f"SWA_EVERY=50",
    f"VE_LAYERS=9,10",                 # match record exactly (was 8,9,10)
    # --- No depth recurrence (record has none; RECUR was never validated) ---
    # --- PRIMARY FIX: WD matching the 1.1147 record ---
    f"MUON_WD=0.04",                   # was 0.090 — the most likely root cause of all failures
    f"EMBED_WD=0.04",                  # was 0.090 — consistent with record
    f"ADAM_WD=0.04",                   # was 0.02 — consistent with record
    # --- No sigmoid skip gates (record doesn't use them) ---
    f"SKIP_GATES_ENABLED=0",           # was 1
    # --- LAWA (our addition on top of record config) ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- No Mousse (record doesn't use it; stripped to match record) ---
    f"MOUSSE_ENABLED=0",               # was 1
    # --- GPTQ calibration (256 seqs = record-level Hessian quality) ---
    f"GPTQ_AR_SEQS=256",
    # --- NO TTT (record: 25 attempts, neutral/negative on Full-GPTQ stack) ---
    f"TTT_ENABLED=0",
    # --- No Z-loss (1.1147 record doesn't use it) ---
    f"Z_LOSS_WEIGHT=0.0",              # was 0.0001
    # --- N-gram tilt ---
    f"NGRAM_BETA=0.5",
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
