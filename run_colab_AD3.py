# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_AD3 Training Run
# **Developed from strategy_AD2 — branch `main`**
#
# ## What AD2 had (all carried forward)
# - MuonEq-R: row-normalise gradient before NS5 inside Muon
# - Skip gates: learnable sigmoid gates on U-Net skip connections
# - Parallel residuals: attention lane + MLP lane merged with learned α
# - Depth recurrence: repeat recur_layers at recur_start_step
# - EMA weight averaging (dynamic decay schedule)
# - Score-first TTT (AdamW + LLRD + OneCycleLR)
# - [AD2-1] RECUR_COUNT — multi-step depth recurrence (3 total passes)
# - [AD2-2] TTT AdamW — AdamW replaces SGD in test-time training
# - [AD2-3] Dynamic EMA — fast early / slow late
# - [AD2-4] TTT LLRD + OneCycleLR schedule per chunk
# - [AD2-5] Adaptive recurrence ramp (+1 pass every recur_ramp_steps)
# - [AD2-6] Per-pass untied adapters — each recurrence pass has its own tiny gated MLP
# - [AD2-7] Differential Attention — softmax(Q1,K1) - λ·softmax(Q2,K2)
#
# ## New in AD3
# - [AD3-fix] forward_logits now correctly unpacks (virtual_layers, pass_nums) and
#             wires per-pass adapter calls in both encoder and decoder phases

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
# ## 2. Dependencies are auto-installed by train_gpt_sota_AD3.py

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42           # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
DATA_DIR = "/kaggle/input/datasets/haphmph/parameter-golf/data"

env = " ".join([
    f"SEED={SEED}",
    f"DATA_DIR={DATA_DIR}",
    f"TARGET_MB={TARGET_MB}",

    # --- Training schedule ---
    f"ITERATIONS=999999",           # stopped by wallclock cap
    f"MAX_WALLCLOCK_SECONDS=4800",  # 600s x 8 GPUs
    f"WARMDOWN_FRAC=0.667",
    f"WARMUP_STEPS=20",
    f"VAL_LOSS_EVERY=4000",
    f"TRAIN_LOG_EVERY=500",

    # --- Architecture (from AD) ---
    f"NUM_LAYERS=11",
    f"MODEL_DIM=512",
    f"NUM_HEADS=8",
    f"NUM_KV_HEADS=4",
    f"VOCAB_SIZE=1024",          # Kaggle dataset is sp1024
    f"QK_GAIN_INIT=5.0",
    f"SKIP_GATES_ENABLED=1",
    f"PARALLEL_START_LAYER=7",
    f"VE_LAYERS=9,10",

    # --- Optimizer (MuonEq-R built in, no Mousse) ---
    f"MATRIX_LR=0.022",
    f"MUON_WD=0.095",
    f"EMBED_WD=0.095",
    f"MUON_MOMENTUM=0.99",
    f"MUON_MOMENTUM_WARMUP_STEPS=1500",

    # --- [AD2-1] Multi-step Depth Recurrence ---
    # RECUR_COUNT=2 → 3 total passes through recur_layers
    # [AD2-5] Ramp: activate 1 pass every RECUR_RAMP_STEPS after RECUR_START_STEP
    f"RECUR_LAYERS=3,4,5",
    f"RECUR_START_STEP=2000",
    f"RECUR_COUNT=2",
    f"RECUR_RAMP_STEPS=500",     # 2000 → 1pass, 2500 → 2passes, 3000 → 3passes

    # --- [AD2-6] Per-pass untied adapters ---
    # Each recurrence pass gets its own tiny gated bottleneck MLP
    # 0 = disable; 32–64 recommended (adds <1% params)
    f"RECUR_ADAPTER_DIM=32",

    # --- [AD2-7] Differential Attention ---
    # Enabled from diff_attn_start_layer onward
    f"DIFF_ATTN=1",
    f"DIFF_ATTN_START_LAYER=4",

    # --- [AD2-3] Dynamic EMA ---
    f"EMA_DECAY=0.9965",
    f"EMA_DECAY_EARLY=0.990",
    f"EMA_DECAY_LATE=0.9990",

    # --- [AD2-2] TTT with AdamW + LLRD + OneCycleLR ---
    f"TTT_ENABLED=1",
    f"TTT_OPTIMIZER=adamw",
    f"TTT_BETA1=0.9",
    f"TTT_BETA2=0.99",
    f"TTT_LR=0.005",
    f"TTT_LLRD=0.85",            # layer-wise LR decay: top=TTT_LR, bottom=TTT_LR*0.85^depth
    f"TTT_SCHEDULE=onecycle",    # warmup 10% then cosine anneal per chunk
    f"TTT_EPOCHS=5",
    f"TTT_CHUNK_TOKENS=32768",
    f"TTT_FREEZE_BLOCKS=0",

    # --- Pre-quant TTT --- DISABLED (uses val data before eval = rules violation)
    f"PREQUANT_TTT_ENABLED=0",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_AD3.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
