# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_20 Training Run
# Built on sota_19 (sota_10 + TTT + N-gram + Hash) with training-time improvements:
# - MTP heads=2 (multi-token prediction aux loss during training, weight=0.1)
# - VE128 extended to layers 8,9,10 (3 layers instead of 2)
# - Depth recurrence extended to layers 2,3,4,5 (4 layers, was 3)
# - Recurrence activated earlier: step 1500 (was 3000)
#
# Eval-time stack (TTT + N-gram + Hash) unchanged from sota_19.

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
    # --- Architecture (from sota_10) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    f"WARMDOWN_ITERS=5500",
    f"GPTQ_AR_SEQS=64",
    # --- Training-time improvements vs sota_19 ---
    f"MTP_NUM_HEADS=2",          # multi-token prediction aux heads during training
    f"MTP_LOSS_WEIGHT=0.1",
    f"VE_LAYERS=8,9,10",         # 3 VE layers (sota_19 uses 9,10 only)
    f"RECUR_LAYERS=2,3,4,5",     # 4 recur layers (sota_19 uses 3,4,5)
    f"RECUR_START_STEP=1500",    # earlier activation (sota_19 uses 3000)
    # --- Legal Score-First TTT (same as sota_19) ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.002",
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    f"TTT_FREEZE_BLOCKS=0",
    # --- N-gram tilt (same as sota_19) ---
    f"NGRAM_BETA=0.5",
    # --- Eval-time hash embedding (same as sota_19) ---
    f"HASH_EMB_SIZE=16384",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_20.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
