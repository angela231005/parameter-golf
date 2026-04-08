# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parameter Golf — SOTA_18–27 Training Runner (branch: ml)
# 10 breakthrough ideas targeting BPB < 1.090 (current SOTA: 1.1147).
#
# | IDEA | Script | Key technique | Expected delta |
# |------|--------|---------------|----------------|
# | 18 | sota_18 | AWQ + Hadamard pre-rotation GPTQ | −0.006~−0.012 |
# | 19 | sota_19 | 4-gram + 5-gram hash embedding | −0.002~−0.004 |
# | 20 | sota_20 | int5 mixed-precision GPTQ | −0.003~−0.007 |
# | 21 | sota_21 | recur_passes=3 + untied adapters | −0.003~−0.005 |
# | 22 | sota_22 | Cautious WD + per-layer LR decay | −0.002~−0.003 |
# | 23 | sota_23 | Differential Attention + XSA | −0.003~−0.006 |
# | 24 | sota_24 | 3-lane parallel residual | −0.002~−0.004 |
# | 25 | sota_25 | Self-distillation QAT (fp32 teacher) | −0.004~−0.008 |
# | 26 | sota_26 | Mini 2-expert MoE MLP | −0.002~−0.005 |
# | 27 | sota_27 | Legal TTT + hash embed adapt | −0.003~−0.005 |

# %% [markdown]
# ## 1. Clone repo (branch ml)

# %%
import torch
import glob
import os

REPO_URL = "https://github.com/angela231005/parameter-golf"
REPO_DIR = "parameter-golf"
BRANCH   = "ml"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} fetch origin")
    os.system(f"git -C {REPO_DIR} checkout {BRANCH}")
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown]
# ## 2. Install dependencies

# %%
os.system("pip install -q sentencepiece zstandard brotli")
os.system('python3 -c "import sentencepiece, zstandard, brotli; print(\'deps OK\')"')

# %% [markdown]
# ## 3. Set hyperparameters

# %%
# --- Tune these ---
IDEA_NUM  = 18          # Which idea to run: 18–27
SEED      = 1337        # change per run: 314, 42, 999
NPROC     = torch.cuda.device_count()  # auto-detect; override manually if needed
TARGET_MB = 15.9

# --- Paths ---
DATA_PATH      = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

ITERATIONS = 6927

# Step milestones: scale proportionally with ITERATIONS
# so QAT/recurrence/warmdown ratios stay correct regardless of total steps.
_qat_step   = int(ITERATIONS * 0.29)   # ~29% → LR starts warmdown → best QAT window
_recur_step = int(ITERATIONS * 0.20)   # ~20% → enough pretraining before recur cost
_warmdown   = int(ITERATIONS * 0.57)   # ~57% matches sota_17 (4000/6927)

# --- Per-idea extra env vars ---
IDEA_EXTRA = {
    18: [  # AWQ + Hadamard GPTQ
        "HADAMARD_ROTATION=1",
        "AWQ_ALPHA=0.5",
        "GPTQ_AR_SEQS=64",
        # Scale QAT to begin during warmdown (not at LR peak)
        f"QAT_START_STEP={_qat_step}",
    ],
    19: [  # 4-gram + 5-gram hash
        "BIGRAM_VOCAB_SIZE=3072",
        "TRIGRAM=1",
    ],
    20: [  # int5 mixed precision
        "MIXED_PRECISION_QUANT=1",
        "BOUNDARY_LAYERS_INT8=1",
    ],
    21: [  # recur_passes=3 + adapters
        "RECUR_PASSES=3",
        "RECUR_ADAPTER_DIM=64",
        "RECUR_START_STEP=1000",
        "RECUR_LAYERS=2,3,4,5",
    ],
    22: [  # Cautious WD + layerLR
        "CAUTIOUS_WD=1",
        "LAYER_LR_DECAY=0.92",
        "MUON_WD=0.05",
    ],
    23: [  # Differential attention
        "DIFF_ATTN=1",
        "DIFF_ATTN_START_LAYER=4",
    ],
    24: [  # 3-lane parallel
        "TRIPLE_LANE=1",
        "TRIPLE_LANE_START=5",
        "LOCAL_ATTN_WINDOW=128",
    ],
    25: [  # Self-distillation QAT
        "DISTILL_ALPHA=0.3",
        "DISTILL_TEMP=4.0",
        "QAT_START_STEP=2000",
    ],
    26: [  # Mini MoE
        "MOE_ENABLED=1",
        "NUM_EXPERTS=2",
    ],
    27: [  # TTT + hash adapt
        "TTT_ENABLED=1",
        "TTT_HASH_ADAPT=1",
        "NGRAM_BETA_DECAY=0.15",
        "TTT_EPOCHS=5",
        "TTT_LR=0.001",
        "TTT_CHUNK_SIZE=16384",
    ],
}

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"ITERATIONS={ITERATIONS}",
    f"WARMDOWN_ITERS={_warmdown}",
    f"RECUR_START_STEP={_recur_step}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
    f"TRAIN_BATCH_TOKENS=786432",
    # --- Architecture (same as sota_17 baseline) ---
    f"NUM_LAYERS=11",
    f"MODEL_DIM=512",
    f"NUM_HEADS=8",
    f"NUM_KV_HEADS=4",
    f"MLP_MULT=3.0",
    f"VOCAB_SIZE=1024",
    f"BIGRAM_VOCAB_SIZE=3072",
    f"BIGRAM_DIM=112",
    f"XSA_LAST_N=11",
    f"ROPE_DIMS=16",
    f"QK_GAIN_INIT=4.0",
    f"LN_SCALE=1",
    f"VE_ENABLED=1",
    f"VE_LAYERS=9,10",
    f"SWA_ENABLED=1",
    f"SWA_EVERY=50",
    f"NGPT=1",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    f"RECUR_LAYERS=2,3,4,5",
    f"TORCHINDUCTOR_COMBO_KERNELS=0",
    # --- Idea-specific ---
    *IDEA_EXTRA.get(IDEA_NUM, []),
])

SCRIPT = f"train_gpt_sota_{IDEA_NUM}.py"
cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} {SCRIPT}"
print(f"IDEA={IDEA_NUM}  NPROC={NPROC}")
print("Command:")
print(cmd)

# %% [markdown]
# ## 4. Train!

# %%
os.system(cmd)
