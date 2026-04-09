# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_33 Training Run
# **Score-First TTT (AdamW) + QK_GAIN=5.0 on sota_32 baseline.**
#
# ## Context
# PR #1477 (1.0822 BPB, SP8192 track) proves that Score-First TTT + QK_GAIN=5.0 + Parallel
# Residuals cleanly beats the record. Our sota_33 ports both to SP1024 on top of sota_32.
#
# ## Why these techs are CLEAN (not SLOT, not Pre-Quant TTT)
# | Technique | Verdict | Reason |
# |---|---|---|
# | Score-First TTT (PR #461) | ✅ LEGAL | Score in inference_mode first, then adapt. Causal. |
# | QK_GAIN=5.0 | ✅ LEGAL | Pure training hyperparameter. |
# | SLOT (PR #1313) | ❌ Retro | Optimizes delta on same tokens it scores (2-pass). |
# | Pre-Quant TTT (PR #1482, #1489) | ❌ Val leak | Adapts weights using fineweb_val_*.bin before GPTQ. |
#
# ### Score-First TTT — our implementation
# `run_legal_ttt()` in train_gpt_sota_28.py uses PR #461 protocol:
# 1. **Score** each chunk in `torch.inference_mode()` — no weight change, log loss
# 2. **Adapt** weights via AdamW after scoring — strictly causal
# No look-ahead, no validation-data training.
#
# PR #1477 full stack: SP8192 + QK5.0 + ScoreFirst TTT (3ep) + Parallel Residuals → 1.0822 BPB
# PR #1413 (no parallel residuals): SP8192 + QK5.0 + ScoreFirst TTT → 1.0828 BPB
#
# ## Stack vs sota_32
# | Change              | sota_32  | sota_33  |
# |---------------------|----------|----------|
# | QK_GAIN_INIT        | 4.0      | **5.0** (PR #1477, #1413: ~0.002–0.005 BPB) |
# | TTT_ENABLED         | 0        | **1** (Score-First, legal, PR #461) |
# | TTT_OPTIMIZER       | sgd      | **adamw** (PR #1440: AdamW > SGD for depth-recurrent) |
# | TTT_LR              | —        | **0.0005** |
# | TTT_EPOCHS          | —        | **3** |
# | TTT_FREEZE_BLOCKS   | —        | **1** (freeze block 0) |
# | EMA_DECAY           | 0.9965   | 0.9965 |
# | RECUR_LAYERS        | 4,5      | 4,5 |
# | WD (all)            | 0.04     | 0.04 |
# | SLOT_ENABLED        | 0        | 0 |
#
# ## Expected outcome
# ~1.08–1.09 BPB (sota_32 baseline + Score-First TTT ~0.002-0.005 + QK5.0 ~0.002-0.005)

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
    f"QK_GAIN_INIT=5.0",               # was 4.0; PR #1477/1413 use 5.0 (+0.002-0.005 BPB)
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
    # --- WD matching the 1.1147 record ---
    f"MUON_WD=0.04",
    f"EMBED_WD=0.04",
    f"ADAM_WD=0.04",
    # --- EMA decay (sota_32 addition) ---
    f"EMA_DECAY=0.9965",
    # --- RECUR (sota_32 addition) ---
    f"RECUR_LAYERS=4,5",
    f"RECUR_START_STEP=3000",
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
    # Scores each chunk in inference_mode first, then adapts — strictly causal.
    # PR #1477 uses 3 epochs; PR #1440 shows AdamW > SGD for depth-recurrent models.
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_LR=0.0005",
    f"TTT_EPOCHS=3",
    f"TTT_FREEZE_BLOCKS=1",
    # --- SLOT disabled ---
    f"SLOT_ENABLED=0",
    # --- Eval-time hash embedding (used by TTT too) ---
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
