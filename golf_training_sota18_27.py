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
# # Parameter Golf — SOTA_18–27 Training Runner
#
# Chạy 10 ý tưởng đột phá từ branch `ml`, mục tiêu BPB < 1.090.
#
# | # | Script | Ý tưởng | BPB dự kiến |
# |---|--------|---------|-------------|
# | 18 | sota_18 | AWQ + Hadamard Pre-Rotation GPTQ | -0.006~-0.012 |
# | 19 | sota_19 | 4-gram + 5-gram Hash Embedding | -0.002~-0.004 |
# | 20 | sota_20 | int5 Mixed-Precision GPTQ | -0.003~-0.007 |
# | 21 | sota_21 | recur_passes=3 + Untied Adapters | -0.003~-0.005 |
# | 22 | sota_22 | Cautious WD + Per-Layer LR Decay | -0.002~-0.003 |
# | 23 | sota_23 | Differential Attention + XSA | -0.003~-0.006 |
# | 24 | sota_24 | 3-Lane Parallel Residual | -0.002~-0.004 |
# | 25 | sota_25 | Self-Distillation QAT | -0.004~-0.008 |
# | 26 | sota_26 | Mini 2-Expert MoE MLP | -0.002~-0.005 |
# | 27 | sota_27 | TTT + Hash Embed Adapt + N-gram Beta | -0.003~-0.005 |

# %% [markdown]
# ## 1. Setup — Clone repo (branch ml) + Install deps

# %%
import os
import sys
import torch

# ── EDIT THESE ────────────────────────────────────────────────────────────────
REPO_URL   = "https://github.com/angela231005/parameter-golf"
REPO_DIR   = "parameter-golf"
BRANCH     = "ml"                    # our new branch with ideas 18-27

# Which idea to run (18–27). Set via env var IDEA_NUM or override here.
IDEA_NUM   = int(os.environ.get("IDEA_NUM", "18"))

# Compute resources
NPROC      = int(os.environ.get("NPROC", "8"))          # GPUs per node
SEED       = int(os.environ.get("SEED", "1337"))        # change per run: 42, 999
TARGET_MB  = float(os.environ.get("TARGET_MB", "15.9")) # artifact size limit
# ──────────────────────────────────────────────────────────────────────────────

# Data paths (Kaggle / custom env)
DATA_PATH       = os.environ.get(
    "DATA_PATH",
    "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH  = os.environ.get(
    "TOKENIZER_PATH",
    "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")

assert 18 <= IDEA_NUM <= 27, f"IDEA_NUM must be 18-27, got {IDEA_NUM}"

SCRIPT = f"train_gpt_sota_{IDEA_NUM}.py"
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Running: {SCRIPT}  |  SEED={SEED}  |  nproc={NPROC}  |  target={TARGET_MB}MB")

# %% [markdown]
# ## 2. Clone / Update repo (branch ml)

# %%
if not os.path.exists(REPO_DIR):
    os.system(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} fetch origin")
    os.system(f"git -C {REPO_DIR} checkout {BRANCH}")
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# Verify the script exists
if not os.path.exists(SCRIPT):
    raise FileNotFoundError(
        f"{SCRIPT} not found in {os.getcwd()}. "
        f"Run generate_sota_variants.py on branch {BRANCH} first.")
print(f"Script found: {SCRIPT}  ({os.path.getsize(SCRIPT)//1024} KB)")

# %% [markdown]
# ## 3. Install Dependencies

# %%
os.system("pip install -q sentencepiece zstandard brotli")
os.system("pip install -q flash_attn_3 --find-links "
          "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 || true")
os.system('python3 -c "import sentencepiece, zstandard; print(\'deps OK\')"')

# %% [markdown]
# ## 4. Per-Idea Hyperparameters
#
# Each cell block shows the recommended hyperparameters for that idea.
# Only the block matching IDEA_NUM is used.

# %%
# Base hyperparameters (shared across all ideas)
BASE_ENV = {
    "SEED":               str(SEED),
    "DATA_PATH":          DATA_PATH,
    "TOKENIZER_PATH":     TOKENIZER_PATH,
    "TARGET_MB":          str(TARGET_MB),
    "ITERATIONS":         "6927",
    "WARMDOWN_ITERS":     "4000",
    "TRAIN_SEQ_LEN":      "2048",
    "EVAL_SEQ_LEN":       "2048",
    "TRAIN_BATCH_TOKENS": "786432",
    "NUM_LAYERS":         "11",
    "MODEL_DIM":          "512",
    "NUM_HEADS":          "8",
    "NUM_KV_HEADS":       "4",
    "MLP_MULT":           "3.0",
    "VOCAB_SIZE":         "1024",
    "BIGRAM_VOCAB_SIZE":  "3072",
    "BIGRAM_DIM":         "112",
    "XSA_LAST_N":         "11",
    "ROPE_DIMS":          "16",
    "QK_GAIN_INIT":       "4.0",
    "LN_SCALE":           "1",
    "VE_ENABLED":         "1",
    "VE_LAYERS":          "9,10",
    "SWA_ENABLED":        "1",
    "SWA_EVERY":          "50",
    "NGPT":               "1",
    "PARALLEL_RESIDUAL":  "1",
    "PARALLEL_START_LAYER": "5",
    "RECUR_LAYERS":       "2,3,4,5",
    "MAX_WALLCLOCK_SECONDS": "0",
    "TORCHINDUCTOR_COMBO_KERNELS": "0",
}

# Idea-specific overrides
IDEA_OVERRIDES = {
    18: {  # AWQ + Hadamard GPTQ
        "HADAMARD_ROTATION": "1",
        "AWQ_CALIBRATION":   "1",
        "AWQ_ALPHA":         "0.5",
        "GPTQ_AR_SEQS":      "64",
    },
    19: {  # 4-gram + 5-gram Hash
        "BIGRAM_VOCAB_SIZE": "3072",
        "TRIGRAM":           "1",
        # fourgram/fivegram auto-enabled when vocab_size >= 2048
    },
    20: {  # int5 Mixed Precision
        "MIXED_PRECISION_QUANT": "1",
        "BOUNDARY_LAYERS_INT8":  "1",
    },
    21: {  # recur_passes=3 + adapters
        "RECUR_PASSES":       "3",
        "RECUR_ADAPTER_DIM":  "64",
        "RECUR_START_STEP":   "1000",
        "RECUR_LAYERS":       "2,3,4,5",
    },
    22: {  # Cautious WD + LayerLR
        "CAUTIOUS_WD":    "1",
        "LAYER_LR_DECAY": "0.92",
        "MUON_WD":        "0.05",
    },
    23: {  # Differential Attention
        "DIFF_ATTN":             "1",
        "DIFF_ATTN_START_LAYER": "4",
    },
    24: {  # 3-Lane Parallel
        "TRIPLE_LANE":         "1",
        "TRIPLE_LANE_START":   "5",
        "LOCAL_ATTN_WINDOW":   "128",
    },
    25: {  # Self-Distillation QAT
        "DISTILL_ALPHA":   "0.3",
        "DISTILL_TEMP":    "4.0",
        "QAT_START_STEP":  "2000",
    },
    26: {  # Mini MoE
        "MOE_ENABLED":  "1",
        "NUM_EXPERTS":  "2",
    },
    27: {  # TTT + Hash Adapt
        "TTT_ENABLED":      "1",
        "TTT_HASH_ADAPT":   "1",
        "NGRAM_BETA_DECAY": "0.15",
        "TTT_EPOCHS":       "5",
        "TTT_LR":           "0.001",
        "TTT_CHUNK_SIZE":   "16384",
    },
}

env_dict = {**BASE_ENV, **IDEA_OVERRIDES.get(IDEA_NUM, {})}
env_str = " ".join(f"{k}={v}" for k, v in env_dict.items())
print(f"\n[IDEA-{IDEA_NUM}] Environment variables:")
for k, v in sorted(IDEA_OVERRIDES.get(IDEA_NUM, {}).items()):
    print(f"  {k}={v}  (override)")

# %% [markdown]
# ## 5. Compose & Print Training Command

# %%
cmd = (
    f"{env_str} "
    f"torchrun --standalone --nproc_per_node={NPROC} {SCRIPT}"
)
print("\n" + "="*80)
print("TRAINING COMMAND:")
print("="*80)
print(cmd)
print("="*80 + "\n")

# %% [markdown]
# ## 6. Run Training
#
# Uncomment the last line to actually execute, or copy the command above.

# %%
# Safety: print first, user confirms
print(f"Starting training for IDEA-{IDEA_NUM}: {SCRIPT}")
print(f"SEED={SEED}, nproc={NPROC}, target={TARGET_MB}MB")

# Run — set DRY_RUN=1 to skip execution
if os.environ.get("DRY_RUN", "0") == "1":
    print("[DRY_RUN] Skipping execution")
else:
    ret = os.system(cmd)
    if ret != 0:
        print(f"\n[WARNING] Training exited with code {ret}")
    else:
        print("\n[OK] Training completed successfully")

# %% [markdown]
# ## 7. Expected Output Files
#
# After training, the following files should appear in the repo directory:
# - `final_model.int6.ptz` — compressed quantized model (~15.9MB)
# - `final_model.pt` — full-precision model (large, for debugging)
#
# The key metric logged at the end:
# ```
# final_int6_sliding_window val_bpb: X.XXXX
# ```
# **Target: < 1.090 BPB**

# %%
# Check output file
model_file = "final_model.int6.ptz"
if os.path.exists(model_file):
    size_mb = os.path.getsize(model_file) / 1024 / 1024
    print(f"Model artifact: {model_file}  ({size_mb:.2f} MB)")
    if size_mb > 16.0:
        print("WARNING: artifact exceeds 16MB limit!")
    elif size_mb > TARGET_MB:
        print(f"WARNING: artifact ({size_mb:.2f}MB) exceeds TARGET_MB={TARGET_MB}")
    else:
        print(f"OK: artifact fits in {TARGET_MB}MB budget")
else:
    print(f"No output file yet: {model_file}")

# %% [markdown]
# ## 8. Quick Results Summary
#
# Tổng hợp kết quả sau mỗi seed run:

# %%
# Parse log for BPB results (run per seed separately)
import glob
import re

log_pattern = "*.log"
logs = sorted(glob.glob(log_pattern))
if not logs:
    print("No log files found yet")
else:
    print(f"Found {len(logs)} log file(s):")
    bpbs = []
    for log_f in logs:
        content = open(log_f).read()
        # Extract sliding window BPB
        matches = re.findall(r"final_int6_sliding_window val_bpb:([\d.]+)", content)
        if matches:
            bpb = float(matches[-1])
            bpbs.append(bpb)
            print(f"  {log_f}: sliding_BPB = {bpb:.4f}")
        else:
            # Try non-sliding
            matches2 = re.findall(r"val_bpb:([\d.]+)", content)
            if matches2:
                bpb = float(matches2[-1])
                print(f"  {log_f}: BPB = {bpb:.4f} (non-sliding)")
    if bpbs:
        mean_bpb = sum(bpbs) / len(bpbs)
        print(f"\nMean BPB ({len(bpbs)} seeds): {mean_bpb:.4f}")
        if mean_bpb < 1.09:
            print("TARGET ACHIEVED! BPB < 1.090")
        elif mean_bpb < 1.1147:
            print(f"SOTA BEATEN! (prev: 1.1147, ours: {mean_bpb:.4f})")
        else:
            print(f"Not yet beating SOTA (1.1147). Delta: {mean_bpb - 1.1147:+.4f}")
