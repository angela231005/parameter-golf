# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_35 Training Run
# **Early Loop Activation (35%) + TTT_EPOCHS=5 + Runtime Optimizations.**
#
# ## What changed vs sota_34 (and why)
#
# ### Change 1: RECUR_START_STEP 3000 → 2425 (43% → 35%)
# PR #1420 swept {0.30, 0.35, 0.40, 0.50} and found 0.35 optimal.
# 0.35 × 6927 iterations = ~2425 steps.
# Our sota_32/33/34 used 3000 (≈43%), which is suboptimal — too late to activate loops.
# Below 0.35 the model doesn't get enough non-looped warmup and quality degrades.
#
# ### Change 2: TTT_EPOCHS 3 → 5
# More TTT adaptation rounds per 32K-token chunk.
# PR #1437 (best clean SP8192) uses 3 epochs. PR #1489 (SP1024, illegal) uses 6 epochs.
# 5 epochs is an unexplored middle ground — more adaptation without the time risk of 6.
#
# ### Change 3: TTT_CHUNK_SIZE 32768 → 65536 (runtime optimization)
# Double chunk size → half as many chunks → half the Python loop overhead per TTT pass.
# chunk_seqs = 65536 // 2048 = 32 seqs/chunk (was 16). batch_seqs stays capped at 8.
# Total FLOPs unchanged, but 50% fewer optimizer.step() calls and Python iterations.
# Adaptation frequency halved — compensated by 5 epochs (more thorough per-chunk learning).
#
# ### Change 4: GPTQ_AR_SEQS 256 → 128 (runtime optimization)
# Halve GPTQ calibration time. PR #1394 (1.0856 BPB reference) uses just 32.
# 128 is still 4× the default, giving good Hessian estimates without excessive cost.
#
# ## What did NOT change vs sota_34
# All 4 core fixes from sota_34 are preserved:
#   - TTT_LR=0.005 (10× fix from sota_33)
#   - TTT_FREEZE_BLOCKS=0
#   - NGRAM_BETA=0.5 (Token-Only N-gram Tilt)
#   - RECUR_LAYERS=3,4,5 (3-layer recurrence = 14 virtual layers)
#
# ## Architecture comparison
# | Approach | Virtual Layers | Reference |
# |---|---|---|
# | sota_34 (3-layer 1x) | 14 virtual | PR #1437: 1.0809 BPB |
# | PR #1420 (2-layer 3x) | 17 virtual | PR #1420: 1.0831 BPB (worse!) |
# 14-virtual wins over 17-virtual when causal-corrected → keep RECUR_LAYERS=3,4,5.
#
# ## Reference PRs
# | PR    | BPB     | Stack |
# |-------|---------|-------|
# | #1437 | 1.0809  | ParRes+3L_RECUR(35%)+NGram+TTT |
# | #1460 | 1.0827  | TTT(SGD)+HashEmb(16384) |
# | #1413 | 1.0828  | QK5+TTT(LR=0.005, freeze=0) |

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
    f"QK_GAIN_INIT=5.0",
    f"BIGRAM_VOCAB_SIZE=3072",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    # --- 3-Layer Depth Recurrence (PR #1437) ---
    f"RECUR_LAYERS=3,4,5",
    # FIX: 2425 = 35% of 6927 (PR #1420 ablation: 35% > 40% > 50% > 30%)
    # sota_34 used 3000 (43%) which is suboptimal per PR #1420
    f"RECUR_START_STEP=2425",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    f"SWA_ENABLED=1",
    f"SWA_EVERY=50",
    f"VE_LAYERS=9,10",
    # --- WD ---
    f"MUON_WD=0.04",
    f"EMBED_WD=0.04",
    f"ADAM_WD=0.04",
    # --- EMA ---
    f"EMA_DECAY=0.9965",
    # --- Mousse optimizer ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- GPTQ calibration ---
    # 128 = 4× default (32). Halves calibration vs sota_34's 256. PR #1394 uses 32.
    f"GPTQ_AR_SEQS=128",
    # --- Score-First TTT (all fixes from sota_34 preserved) ---
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_LR=0.005",
    f"TTT_EPOCHS=5",          # 5 epochs × 32 seqs/chunk (65536 chunks)
    f"TTT_CHUNK_SIZE=65536",  # 2× chunk → 50% fewer loop iterations, same total FLOPs
    f"TTT_FREEZE_BLOCKS=0",
    # --- Token-Only N-gram Tilt (PR #1437) ---
    f"NGRAM_BETA=0.5",
    # --- SLOT disabled ---
    f"SLOT_ENABLED=0",
    # --- Eval-time hash embedding ---
    f"HASH_EMB_SIZE=32768",
    # --- Markov curriculum ---
    f"RAKI_POWER=0.10",
    # --- Late QAT ---
    f"LATE_QAT_STEPS=200",
    f"LATE_QAT_THRESHOLD=0",
    # --- NS steps ---
    f"MUON_BACKEND_STEPS=5",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_28.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
