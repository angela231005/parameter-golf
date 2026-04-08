# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_28 Training Run
# Built on sota_26. **Significant changes: Z-loss + Sequence-Length Curriculum + 3-layer RECUR**
#
# **Changes vs sota_27 (train_gpt_sota_28.py — new code changes):**
# - **Z-loss regularization** (`Z_LOSS_WEIGHT=1e-4`, Gemma/PaLM style):
#     - `main_loss += z_loss_weight * logits.logsumexp(-1).pow(2).mean()`
#     - Penalises large log-partition-function → prevents logit magnitude explosion
#     - Tanh softcap limits peak logits, but z-loss regularises the full distribution
#     - Zero extra parameters; only active during training (not eval/TTT)
# - **Sequence-length curriculum** (`SEQ_CURRICULUM="1500,3000"`, `SEQ_CURRICULUM_LENS="512,1024"`):
#     - Steps 0→1500: seq_len=512 (noisier gradients, faster exploration of token patterns)
#     - Steps 1500→3000: seq_len=1024 (medium-range context)
#     - Steps 3000→6927: seq_len=2048 (full long-range context)
#     - DistributedTokenLoader reinits coprime-stride pipeline when seq_len changes
# - **RECUR_LAYERS=3,4,5** (was 4,5) — one extra recurrent layer for free depth
#     - RECUR_START_STEP=2000 (was 2424) — start recurrence earlier
#
# **Hyperparameter changes vs sota_27 (same train_gpt_sota_26.py hparams):**
# - WARMDOWN_ITERS=4000 (inherited from sota_27)
# - SWA_ENABLED=0 (inherited from sota_27)
#
# **Inherited unchanged from sota_26/27:**
# - Sigmoid skip_gates on U-Net skip connections
# - Raki v6 weight decay: MUON_WD=0.090, EMBED_WD=0.090, ADAM_WD=0.02
# - Raki coprime-stride DistributedTokenLoader
# - Markov curriculum (RAKI_POWER=0.10)
# - Late QAT: last 200 steps + dynamo reset
# - Mousse EMA optimizer (beta=0.95), TTT AdamW lr=0.0003
# - LAWA (k=15, freq=50), GPTQ 128 AR seqs, Hash embed 32768
# - PARALLEL_START_LAYER=7, NO MTP, BigramHash 3072×112, XSA all 11 layers

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
    f"WARMDOWN_ITERS=4000",           # from sota_27: record uses 4000
    f"SWA_ENABLED=0",                 # from sota_27: LAWA is better
    f"VE_LAYERS=8,9,10",
    # --- Depth Recurrence (3 layers now, earlier start) ---
    f"RECUR_LAYERS=3,4,5",            # was 4,5: one extra recur layer for free depth
    f"RECUR_START_STEP=2000",         # was 2424: earlier start
    f"RECUR_COUNT=1",
    # --- Z-loss regularization (NEW) ---
    f"Z_LOSS_WEIGHT=0.0001",          # 1e-4: penalise logit magnitude during training
    # --- Sequence-length curriculum (NEW) ---
    f"SEQ_CURRICULUM=1500,3000",      # step thresholds for seq_len transitions
    f"SEQ_CURRICULUM_LENS=512,1024",  # seq_len at each phase (then train_seq_len=2048)
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

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_28.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
