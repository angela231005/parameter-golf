# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_36 Training Run
# **All sota_35 settings + flash-attn library for faster attention.**
#
# ## What changed vs sota_35
#
# ### Change: pip install flash-attn (native GQA, no K/V head expansion)
# The training code previously used PyTorch SDPA with `repeat_interleave` to expand
# 4 KV heads → 8 Q heads before calling SDPA. This copies K/V tensors in memory.
#
# flash-attn 2.x handles GQA natively (num_heads_k != num_heads_q) without any
# expansion. The library is registered as a torch.library op, so it is compatible
# with `torch.compile(fullgraph=True)`.
#
# Benefits:
#   - Removes K/V repeat_interleave copy (saves ~2× K/V memory bandwidth per layer)
#   - flash-attn 2 kernel is typically faster than SDPA flash backend on A100/H100
#   - Applies to BOTH main training and TTT eval (both go through flash_attn_3_func)
#   - If flash-attn is unavailable (install failure), train_gpt_sota_28.py falls
#     back to the SDPA path automatically — run is not broken.
#
# ## What did NOT change vs sota_35
# All hyperparameters are identical:
#   - RECUR_START_STEP=2425 (35%)
#   - TTT_EPOCHS=5, TTT_CHUNK_SIZE=65536, TTT_LR=0.005
#   - GPTQ_AR_SEQS=128
#   - NGRAM_BETA=0.5, RECUR_LAYERS=3,4,5, TTT_FREEZE_BLOCKS=0
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

# Install flash-attn: native GQA, torch.compile fullgraph compatible (requires 2.6+).
# --no-build-isolation: use pre-built wheels where available (fast install on Kaggle).
os.system("pip install flash-attn --no-build-isolation -q")
os.system('python3 -c "import flash_attn; print(\'flash_attn\', flash_attn.__version__)"')

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
    # 2425 = 35% of 6927 (PR #1420 ablation: 35% optimal)
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
    f"GPTQ_AR_SEQS=128",
    # --- Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_LR=0.005",
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_SIZE=65536",
    f"TTT_FREEZE_BLOCKS=0",
    # --- Token-Only N-gram Tilt ---
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
