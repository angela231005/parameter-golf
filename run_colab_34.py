# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_34 Training Run
# **4 key fixes on sota_33: correct TTT_LR, unfrozen TTT, N-gram Tilt, 3-layer RECUR.**
#
# ## What changed vs sota_33 (and why)
#
# ### Fix 1: TTT_LR 0.0005 → 0.005 (critical 10× bug)
# PR #1413 (dexhunter, 1.0828 BPB) uses TTT_LR=0.005. Our sota_33 used 0.0005 — 10×
# weaker adaptation. This is the most impactful single fix.
#
# ### Fix 2: TTT_FREEZE_BLOCKS 1 → 0 (unfreeze all blocks)
# PR #1413 uses freeze_blocks=0 (all 11 blocks adapt during TTT). sota_33 froze block 0.
# Letting the embedding block also adapt during TTT should give slightly better BPB.
#
# ### Fix 3: NGRAM_BETA=0.5 (enable Token-Only N-gram Tilt)
# PR #1437 (1.0809 BPB) includes causal token-only N-gram Tilt. Our implementation in
# run_legal_ttt() is already correct:
#   p_tilt(t) = p_model(t) * exp(beta * 1[t==bigram_hint(prev)]) / Z
# Bigram table uses ONLY prefix tokens (x_{t-1}), updated AFTER scoring (causal).
# This gives ~-0.00014 BPB improvement (small but free, zero extra params).
#
# ### Fix 4: RECUR_LAYERS=3,4,5 (3-layer recurrence, not 2-layer)
# PR #1437 (1.0809 BPB) uses LOOP_START=3, LOOP_END=5 (layers 3,4,5).
# sota_32/33 used only RECUR_LAYERS=4,5. The default in train_gpt_sota_28.py is already
# "3,4,5" — sota_33 explicitly set this to "4,5", which was wrong.
#
# ## Stack vs sota_33
# | Change              | sota_33         | sota_34                        |
# |---------------------|-----------------|-------------------------------|
# | TTT_LR              | 0.0005 ❌      | **0.005** (PR #1413)           |
# | TTT_FREEZE_BLOCKS   | 1 ❌            | **0** (PR #1413)               |
# | NGRAM_BETA          | not set (0.0) ❌| **0.5** (PR #1437 token-only)  |
# | RECUR_LAYERS        | 4,5 ❌          | **3,4,5** (PR #1437)           |
# | QK_GAIN_INIT        | 5.0 ✅          | 5.0                            |
# | TTT_ENABLED         | 1 ✅            | 1                              |
# | TTT_OPTIMIZER       | adamw ✅        | adamw                          |
# | TTT_EPOCHS          | 3 ✅            | 3                              |
# | EMA_DECAY           | 0.9965 ✅       | 0.9965                         |
# | PARALLEL_RESIDUAL   | 1 (layers 7+)✅ | 1 (layers 7+)                  |
#
# ## Reference PRs (SP8192, all clean/causal)
# | PR    | BPB     | Stack |
# |-------|---------|-------|
# | #1413 | 1.0828  | QK5+TTT(LR=0.005, freeze=0) |
# | #1437 | 1.0809  | ParRes+3L_RECUR+NGram_Tilt+TTT |
# | #1477 | 1.0822  | ParRes+TTT(3ep) |
#
# ## Expected outcome
# ~1.075–1.083 BPB. The TTT_LR fix alone should recover most of the gap from sota_33.
# N-gram Tilt adds a small but real additional improvement.

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
    f"QK_GAIN_INIT=5.0",               # PR #1477/1413: QK gain boost
    f"BIGRAM_VOCAB_SIZE=3072",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals (PR #1412, #1437) ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",         # layers 7-10 (decoder), matching PR #1437
    # --- 3-Layer Depth Recurrence (PR #1437) ---
    # PR #1437 uses LOOP_START=3, LOOP_END=5 = 3 physical layers {3,4,5}
    # Default in train_gpt_sota_28.py is already "3,4,5" — sota_33 wrongly set to "4,5"
    f"RECUR_LAYERS=3,4,5",
    f"RECUR_START_STEP=3000",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=4000",
    f"SWA_ENABLED=1",
    f"SWA_EVERY=50",
    f"VE_LAYERS=9,10",
    # --- WD matching the 1.1147 record ---
    f"MUON_WD=0.04",
    f"EMBED_WD=0.04",
    f"ADAM_WD=0.04",
    # --- EMA decay (sota_32 addition) ---
    f"EMA_DECAY=0.9965",
    # --- Mousse optimizer ---
    f"MOUSSE_ENABLED=1",
    f"MOUSSE_BETA=0.95",
    # --- LAWA ---
    f"LAWA_ENABLED=1",
    f"LAWA_K=15",
    f"LAWA_FREQ=50",
    # --- GPTQ calibration ---
    f"GPTQ_AR_SEQS=256",
    # --- Score-First TTT (legal, PR #461 protocol) ---
    # CRITICAL FIXES vs sota_33:
    #   TTT_LR: 0.0005 → 0.005 (PR #1413 uses 0.005, was 10x too weak)
    #   TTT_FREEZE_BLOCKS: 1 → 0 (PR #1413 unfreezes all blocks)
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_LR=0.005",                   # FIX: was 0.0005 in sota_33 (10x wrong)
    f"TTT_EPOCHS=3",
    f"TTT_FREEZE_BLOCKS=0",            # FIX: was 1 in sota_33; PR #1413 uses 0
    # --- Token-Only N-gram Tilt (PR #1437) ---
    # Formula: p_tilt(t) = p_model(t) * exp(beta * 1[t==bigram_hint(x_{t-1})]) / Z
    # Causal: hint from x_{t-1} only; bigram table updated AFTER scoring each chunk.
    # Our run_legal_ttt() implementation is token-only by default (no within/word experts).
    f"NGRAM_BETA=0.5",                 # NEW: enable N-gram Tilt; default was not set (0.0)
    # --- SLOT disabled ---
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
