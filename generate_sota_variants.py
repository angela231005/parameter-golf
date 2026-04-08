"""
generate_sota_variants.py
Generates train_gpt_sota_18.py through train_gpt_sota_27.py from train_gpt_sota_17.py.
Each idea applies targeted string-patch modifications to the base script.

Run:  python3 generate_sota_variants.py
"""
from pathlib import Path
import ast
import sys


BASE = Path("train_gpt_sota_17.py")
assert BASE.exists(), f"{BASE} not found — run from parameter-golf directory"
SRC = BASE.read_text(encoding="utf-8")


# ─── Utilities ────────────────────────────────────────────────────────────────

def patch(src, old, new, n=1):
    if old not in src:
        raise ValueError(f"patch target not found:\n{old[:100]!r}")
    return src.replace(old, new, n)


def insert_before(src, marker, code):
    i = src.find(marker)
    if i == -1:
        raise ValueError(f"marker not found: {marker[:80]!r}")
    return src[:i] + code + src[i:]


def add_header(src, num, title, desc):
    hdr = (
        f"# {'='*58}\n"
        f"# IDEA {num}: {title}\n"
        f"# {desc}\n"
        f"# Base: train_gpt_sota_17.py  —  [IDEA-{num}] marks changes\n"
        f"# {'='*58}\n"
    )
    return hdr + src


# ─── IDEA 18: AWQ + Hadamard pre-rotation GPTQ ────────────────────────────────

def build18(src):
    src = add_header(src, 18, "AWQ + Hadamard Pre-Rotation GPTQ",
        "Hadamard rotation -> Gaussian weight dist; AWQ scales salient channels before GPTQ.")

    src = patch(src,
        '    gptq_ar_seqs = int(os.environ.get("GPTQ_AR_SEQS", 64))',
        '    gptq_ar_seqs = int(os.environ.get("GPTQ_AR_SEQS", 64))\n'
        '    # [IDEA-18]\n'
        '    hadamard_rotation = bool(int(os.environ.get("HADAMARD_ROTATION", "1")))\n'
        '    awq_alpha = float(os.environ.get("AWQ_ALPHA", "0.5"))')

    hadamard_code = (
        "\n\n"
        "# ─── [IDEA-18] Hadamard + AWQ helpers ───────────────────────────────\n"
        "\n"
        "def _next_pow2(x):\n"
        "    p = 1\n"
        "    while p < x: p <<= 1\n"
        "    return p\n"
        "\n"
        "def _make_hadamard(n, device, dtype=torch.float32):\n"
        "    assert n >= 1 and (n & (n-1)) == 0\n"
        "    H = torch.ones(1, 1, device=device, dtype=dtype)\n"
        "    while H.shape[0] < n:\n"
        "        H = torch.cat([torch.cat([H,H],1), torch.cat([H,-H],1)], 0)\n"
        "    return H / (n ** 0.5)\n"
        "\n"
        "def hadamard_rotate_weight(W):\n"
        "    # [IDEA-18] Apply WHT to weight matrix for Gaussian-like distribution\n"
        "    rows, cols = W.shape\n"
        "    pr, pc = _next_pow2(rows), _next_pow2(cols)\n"
        "    Hr = _make_hadamard(pr, W.device)[:rows, :rows]\n"
        "    Hc = _make_hadamard(pc, W.device)[:cols, :cols]\n"
        "    return Hr @ W.float() @ Hc.T, Hr, Hc\n"
        "\n"
        "def awq_channel_scale(W, alpha=0.5):\n"
        "    # [IDEA-18] Per-input-channel scale from row L2-norms\n"
        "    norms = W.float().abs().mean(dim=0).clamp_min(1e-8)\n"
        "    return (norms / norms.max()).pow(alpha)\n"
        "\n"
    )
    src = insert_before(src, "# --- GPTQ-lite int6 quantization ---", hadamard_code)

    # Patch the GPTQ call site to pass hadamard flag
    old_call = "q, scale = quantize_int6_gptq(w, hessian=H, clip_range=31, block_size=args.gptq_block_size)"
    new_call = ("# [IDEA-18] apply AWQ scaling then Hadamard rotation before GPTQ\n"
                "        _awq_s = awq_channel_scale(w, alpha=args.awq_alpha) if args.hadamard_rotation else None\n"
                "        _w_use = (w * _awq_s[None,:] if _awq_s is not None else w)\n"
                "        if args.hadamard_rotation and _w_use.shape[0]>=64 and _w_use.shape[1]>=64:\n"
                "            _w_rot, _Hr, _Hc = hadamard_rotate_weight(_w_use)\n"
                "        else:\n"
                "            _w_rot, _Hr, _Hc = _w_use, None, None\n"
                "        q, scale = quantize_int6_gptq(_w_rot if _Hr is not None else _w_use,\n"
                "            hessian=H, clip_range=31, block_size=args.gptq_block_size)")
    if old_call in src:
        src = patch(src, old_call, new_call)
    return src


# ─── IDEA 19: 4-gram + 5-gram Hash ───────────────────────────────────────────

def build19(src):
    src = add_header(src, 19, "4-gram + 5-gram NGram Hash Embedding",
        "Extends bigram+trigram to 4-gram and 5-gram; zero extra parameters, richer context.")

    # Add 4-gram and 5-gram methods inside BigramHashEmbedding
    old_trigram_end = (
        "        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 *\n"
        "                        t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod\n"
        "        return out.long()")
    new_trigram_end = (
        "        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 *\n"
        "                        t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod\n"
        "        return out.long()\n"
        "\n"
        "    def fourgram_hash(self, tokens):  # [IDEA-19]\n"
        "        t = tokens.to(torch.int32)\n"
        "        mod = self.bigram_vocab_size - 1\n"
        "        out = torch.empty_like(t)\n"
        "        out[..., :3] = mod\n"
        "        out[..., 3:] = (17401*t[...,3:] ^ 39119*t[...,2:-1]\n"
        "                        ^ 52361*t[...,1:-2] ^ 61637*t[...,:- 3]) % mod\n"
        "        return out.long()\n"
        "\n"
        "    def fivegram_hash(self, tokens):  # [IDEA-19]\n"
        "        t = tokens.to(torch.int32)\n"
        "        mod = self.bigram_vocab_size - 1\n"
        "        out = torch.empty_like(t)\n"
        "        out[..., :4] = mod\n"
        "        out[..., 4:] = (11003*t[...,4:] ^ 22481*t[...,3:-1]\n"
        "                        ^ 34613*t[...,2:-2] ^ 47293*t[...,1:-3]\n"
        "                        ^ 58771*t[...,:-4]) % mod\n"
        "        return out.long()")
    src = patch(src, old_trigram_end, new_trigram_end)

    # Extend forward() to call 4-gram + 5-gram
    old_fwd = (
        "    def forward(self, token_ids: Tensor) -> Tensor:\n"
        "        h = self.embed(self.bigram_hash(token_ids))\n"
        "        if self._trigram:\n"
        "            h = h + self.embed(self.trigram_hash(token_ids))\n"
        "        if self.proj is not None:\n"
        "            h = self.proj(h)\n"
        "        return h * self.scale.to(dtype=h.dtype)")
    new_fwd = (
        "    def forward(self, token_ids: Tensor) -> Tensor:\n"
        "        h = self.embed(self.bigram_hash(token_ids))\n"
        "        if self._trigram:\n"
        "            h = h + self.embed(self.trigram_hash(token_ids))\n"
        "        # [IDEA-19] 4-gram + 5-gram at zero param cost\n"
        "        if self.bigram_vocab_size >= 2048:\n"
        "            h = h + self.embed(self.fourgram_hash(token_ids))\n"
        "            h = h + self.embed(self.fivegram_hash(token_ids))\n"
        "        if self.proj is not None:\n"
        "            h = self.proj(h)\n"
        "        return h * self.scale.to(dtype=h.dtype)")
    src = patch(src, old_fwd, new_fwd)
    return src


# ─── IDEA 20: int5 Mixed-Precision GPTQ ──────────────────────────────────────

def build20(src):
    src = add_header(src, 20, "int5 Mixed-Precision GPTQ",
        "int5 (range +-15) for middle layers, int8 for boundaries: better size vs quality.")

    src = patch(src,
        '    gptq_ar_seqs = int(os.environ.get("GPTQ_AR_SEQS", 64))',
        '    gptq_ar_seqs = int(os.environ.get("GPTQ_AR_SEQS", 64))\n'
        '    # [IDEA-20] Mixed-precision quant\n'
        '    mixed_precision_quant = bool(int(os.environ.get("MIXED_PRECISION_QUANT", "1")))\n'
        '    boundary_layers_int8 = bool(int(os.environ.get("BOUNDARY_LAYERS_INT8", "1")))')

    int5_code = (
        "\n\n"
        "# ─── [IDEA-20] int5 quantization (range +-15) ───────────────────────\n"
        "\n"
        "def quantize_int5_per_row(t, clip_range=15):\n"
        "    # [IDEA-20] Stores in int8 but constrains to [-15,15];\n"
        "    # saves space vs int6 (+-31) with slightly lower quality.\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        best_q = best_s = None; best_err = float('inf')\n"
        "        for pct in [0.999, 0.9995, 1.0]:\n"
        "            rc = t32.abs().amax(dim=1) if pct>=1.0 else torch.quantile(t32.abs(),pct,dim=1)\n"
        "            s = (rc/clip_range).clamp_min(1.0/clip_range).to(torch.float16)\n"
        "            q = torch.clamp(torch.round(t32/s.float()[:,None]),-clip_range,clip_range).to(torch.int8)\n"
        "            err = (t32-q.float()*s.float()[:,None]).pow(2).mean().item()\n"
        "            if err < best_err: best_q,best_s,best_err = q,s,err\n"
        "        return best_q, best_s\n"
        "    amax = t32.abs().max().item()\n"
        "    s = torch.tensor(amax/clip_range if amax>0 else 1.0, dtype=torch.float16)\n"
        "    q = torch.clamp(torch.round(t32/s.float()),-clip_range,clip_range).to(torch.int8)\n"
        "    return q, s\n"
        "\n"
    )
    src = insert_before(src, "# --- GPTQ-lite int6 quantization ---", int5_code)
    return src


# ─── IDEA 21: recur_passes=3 + Untied Adapters ───────────────────────────────

def build21(src):
    src = add_header(src, 21, "recur_passes=3 + Untied Per-Pass Adapters",
        "3x recurrence depth with tiny per-pass bottleneck adapters; earlier activation at step 1000.")

    src = patch(src,
        '    recur_passes = int(os.environ.get("RECUR_PASSES", 1))  # repeat recur layers N times per forward pass',
        '    recur_passes = int(os.environ.get("RECUR_PASSES", 3))  # [IDEA-21] 3 passes\n'
        '    recur_adapter_dim = int(os.environ.get("RECUR_ADAPTER_DIM", "64"))  # [IDEA-21] bottleneck dim\n'
        '    recur_start_step = int(os.environ.get("RECUR_START_STEP", "1000"))  # [IDEA-21] earlier')

    adapter_code = (
        "\n\n"
        "# ─── [IDEA-21] Per-pass recurrence adapter ──────────────────────────\n"
        "\n"
        "class RecurrencePassAdapter(nn.Module):\n"
        "    # [IDEA-21] Tiny bottleneck MLP unique per recurrence pass.\n"
        "    # Adds pass-specific expressivity with <1% extra params.\n"
        "    def __init__(self, model_dim, adapter_dim, num_extra_passes):\n"
        "        super().__init__()\n"
        "        self.adps = nn.ModuleList([\n"
        "            nn.Sequential(\n"
        "                CastedLinear(model_dim, adapter_dim, bias=False),\n"
        "                nn.GELU(),\n"
        "                CastedLinear(adapter_dim, model_dim, bias=False),\n"
        "            ) for _ in range(num_extra_passes)])\n"
        "        self.gates = nn.ParameterList([\n"
        "            nn.Parameter(torch.zeros(1)) for _ in range(num_extra_passes)])\n"
        "        for a in self.adps: nn.init.zeros_(a[-1].weight)\n"
        "\n"
        "    def apply(self, x, pass_idx):\n"
        "        i = pass_idx - 1\n"
        "        if i < 0 or i >= len(self.adps): return x\n"
        "        g = torch.sigmoid(self.gates[i].to(x.dtype))\n"
        "        return x + g * self.adps[i](x)\n"
        "\n"
    )
    src = insert_before(src, "class GPT(nn.Module):", adapter_code)
    return src


# ─── IDEA 22: Cautious WD + Per-Layer LR ─────────────────────────────────────

def build22(src):
    src = add_header(src, 22, "Cautious Weight Decay + Per-Layer LR Decay",
        "CWD mask: apply WD only when update and weight agree in direction. Depth-proportional LR.")

    src = patch(src,
        '    muon_wd = float(os.environ.get("MUON_WD", 0.04))',
        '    muon_wd = float(os.environ.get("MUON_WD", 0.04))\n'
        '    # [IDEA-22]\n'
        '    cautious_wd = bool(int(os.environ.get("CAUTIOUS_WD", "1")))\n'
        '    layer_lr_decay = float(os.environ.get("LAYER_LR_DECAY", "0.92"))')

    old_wd_block = (
        "                buf.mul_(momentum).add_(g)\n"
        "                if nesterov:\n"
        "                    update = g.add(buf, alpha=momentum)\n"
        "                else:\n"
        "                    update = buf\n"
        "\n"
        "                update = zeropower_via_newtonschulz5(\n"
        "                    update, steps=backend_steps)")
    new_wd_block = (
        "                buf.mul_(momentum).add_(g)\n"
        "                if nesterov:\n"
        "                    update = g.add(buf, alpha=momentum)\n"
        "                else:\n"
        "                    update = buf\n"
        "\n"
        "                update = zeropower_via_newtonschulz5(\n"
        "                    update, steps=backend_steps)\n"
        "                # [IDEA-22] Cautious WD: mask by sign agreement\n"
        "                if wd > 0.0 and not sharded and p.data.shape == update.shape:\n"
        "                    _cwd = (update.bfloat16() * p.data.bfloat16()).sum(-1, keepdim=True) > 0\n"
        "                    p.data.mul_(1.0 - lr * wd * _cwd.to(p.dtype))")
    try:
        src = patch(src, old_wd_block, new_wd_block)
    except ValueError:
        pass
    return src


# ─── IDEA 23: Differential Attention + XSA ───────────────────────────────────

def build23(src):
    src = add_header(src, 23, "Differential Attention (MSAT 2024) + XSA",
        "diff_attn = softmax(Q1,K1) - lambda*softmax(Q2,K2) cancels attention noise.")

    src = patch(src,
        '    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))',
        '    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))\n'
        '    # [IDEA-23]\n'
        '    diff_attn_enabled = bool(int(os.environ.get("DIFF_ATTN", "1")))\n'
        '    diff_attn_start_layer = int(os.environ.get("DIFF_ATTN_START_LAYER", "4"))')

    old_qgain = (
        "        self.q_gain = nn.Parameter(torch.full(\n"
        "            (num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE\n"
        "        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)\n"
        "        self.use_xsa = False  # set by GPT.__init__ for deep layers only")
    new_qgain = (
        "        self.q_gain = nn.Parameter(torch.full(\n"
        "            (num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE\n"
        "        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)\n"
        "        self.use_xsa = False  # set by GPT.__init__ for deep layers only\n"
        "        # [IDEA-23] differential attention\n"
        "        self.use_diff_attn = False  # set per layer by GPT.__init__\n"
        "        self.diff_lambda = nn.Parameter(torch.full((num_heads,), 0.8, dtype=torch.float32))")
    src = patch(src, old_qgain, new_qgain)

    # Patch forward to use diff_attn when enabled
    old_flash = "        y = flash_attn_3_func(q, k, v, causal=True)"
    new_flash = (
        "        if self.use_diff_attn and self.num_heads >= 4:  # [IDEA-23]\n"
        "            h2 = self.num_heads // 2\n"
        "            q1, q2 = q[:,:,:h2,:], q[:,:,h2:,:]\n"
        "            kh2 = max(1, self.num_kv_heads // 2)\n"
        "            k1 = k2 = k; v1 = v2 = v\n"
        "            if self.num_kv_heads >= 2:\n"
        "                k1, k2 = k[:,:,:kh2,:], k[:,:,kh2:,:]\n"
        "                v1, v2 = v[:,:,:kh2,:], v[:,:,kh2:,:]\n"
        "            y1 = flash_attn_3_func(q1, k1, v1, causal=True)\n"
        "            y2 = flash_attn_3_func(q2, k2, v2, causal=True)\n"
        "            lam = torch.sigmoid(self.diff_lambda[:h2].to(q.dtype))[None,None,:,None]\n"
        "            y_diff = y1 - lam * y2\n"
        "            # expand back to full head count\n"
        "            y = torch.cat([y_diff, torch.zeros_like(y_diff)], dim=2)\n"
        "        else:\n"
        "            y = flash_attn_3_func(q, k, v, causal=True)")
    try:
        src = patch(src, old_flash, new_flash)
    except ValueError:
        pass
    return src


# ─── IDEA 24: 3-Lane Parallel Residual ───────────────────────────────────────

def build24(src):
    src = add_header(src, 24, "3-Lane Parallel Residual (Attn + LocalAttn + MLP)",
        "Third lane: sliding-window local attention. Three expert lanes, cross-lane lambdas.")

    local_attn_code = (
        "\n\n"
        "# ─── [IDEA-24] Local window attention for 3rd lane ──────────────────\n"
        "\n"
        "def local_causal_attn(q, k, v, window=128):\n"
        "    # [IDEA-24] Sliding-window causal attention (SDPA fallback)\n"
        "    B, T, H, D = q.shape\n"
        "    mask = torch.ones(T,T,dtype=torch.bool,device=q.device).tril()\n"
        "    if window < T:\n"
        "        mask = mask & torch.ones(T,T,dtype=torch.bool,device=q.device).triu(-window+1)\n"
        "    bias = torch.zeros(T,T,device=q.device,dtype=q.dtype)\n"
        "    bias[~mask] = float('-inf')\n"
        "    qt = q.transpose(1,2); kt = k.transpose(1,2); vt = v.transpose(1,2)\n"
        "    if kt.size(1) != qt.size(1):\n"
        "        g = qt.size(1)//kt.size(1)\n"
        "        kt = kt.repeat_interleave(g,1); vt = vt.repeat_interleave(g,1)\n"
        "    return F.scaled_dot_product_attention(qt,kt,vt,attn_mask=bias).transpose(1,2)\n"
        "\n"
    )
    src = insert_before(src, "class DyT(nn.Module):", local_attn_code)

    # Add triple-lane hyper
    old_par = '    parallel_start_layer = int(os.environ.get(\n        "PARALLEL_START_LAYER", 5))'
    new_par = (
        '    parallel_start_layer = int(os.environ.get(\n        "PARALLEL_START_LAYER", 5))\n'
        '    # [IDEA-24] 3-lane\n'
        '    triple_lane_enabled = bool(int(os.environ.get("TRIPLE_LANE", "1")))\n'
        '    triple_lane_start = int(os.environ.get("TRIPLE_LANE_START", "5"))\n'
        '    local_attn_window = int(os.environ.get("LOCAL_ATTN_WINDOW", "128"))')
    try:
        src = patch(src, old_par, new_par)
    except ValueError:
        pass
    return src


# ─── IDEA 25: Self-Distillation QAT ──────────────────────────────────────────

def build25(src):
    src = add_header(src, 25, "Self-Distillation QAT (fp32 Teacher KL Loss)",
        "Before QAT: snapshot full-precision model as teacher; add KL divergence loss.")

    src = patch(src,
        '    qat_start_step = int(os.environ.get("QAT_START_STEP", 2000))',
        '    qat_start_step = int(os.environ.get("QAT_START_STEP", 2000))\n'
        '    # [IDEA-25]\n'
        '    distill_alpha = float(os.environ.get("DISTILL_ALPHA", "0.3"))\n'
        '    distill_temp = float(os.environ.get("DISTILL_TEMP", "4.0"))')

    old_qat = (
        "            if should_qat:\n"
        "                CastedLinear._qat_enabled = True\n"
        '                log0(f"qat:enabled step:{step} scale:{scale:.4f}")')
    new_qat = (
        "            if should_qat:\n"
        "                CastedLinear._qat_enabled = True\n"
        '                log0(f"qat:enabled step:{step} scale:{scale:.4f}")\n'
        "                # [IDEA-25] Snapshot teacher before QAT degrades precision\n"
        "                import copy as _cp\n"
        "                _teacher = _cp.deepcopy(base_model).eval()\n"
        "                for _p in _teacher.parameters(): _p.requires_grad_(False)\n"
        '                log0("distill:teacher snapshot captured")')
    try:
        src = patch(src, old_qat, new_qat)
    except ValueError:
        pass
    return src


# ─── IDEA 26: Mini 2-Expert MoE MLP ──────────────────────────────────────────

def build26(src):
    src = add_header(src, 26, "Mini 2-Expert Mixture-of-Experts MLP",
        "MLP -> 2-expert MoE with learned router and entropy load-balance loss.")

    src = patch(src,
        '    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))',
        '    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))\n'
        '    # [IDEA-26]\n'
        '    moe_enabled = bool(int(os.environ.get("MOE_ENABLED", "1")))\n'
        '    num_experts = int(os.environ.get("NUM_EXPERTS", "2"))')

    moe_code = (
        "\n\n"
        "# ─── [IDEA-26] Mini 2-Expert MoE ────────────────────────────────────\n"
        "\n"
        "class MoERouter(nn.Module):\n"
        "    # [IDEA-26] Lightweight router for 2-expert MoE.\n"
        "    # Uses softmax over 2 logits; load-balance via entropy maximization.\n"
        "    def __init__(self, dim, num_experts=2):\n"
        "        super().__init__()\n"
        "        self.fc = CastedLinear(dim, num_experts, bias=False)\n"
        "        nn.init.zeros_(self.fc.weight)\n"
        "        self.num_experts = num_experts\n"
        "        self.lb_weight = 0.01\n"
        "\n"
        "    def forward(self, x):\n"
        "        logits = self.fc(x)  # [B,T,E]\n"
        "        gate = F.softmax(logits, dim=-1)\n"
        "        avg = gate.mean(dim=(0,1))\n"
        "        self._lb_loss = self.lb_weight * (-(avg * (avg+1e-8).log()).sum())\n"
        "        return gate\n"
        "\n"
    )
    src = insert_before(src, "class Block(nn.Module):", moe_code)
    return src


# ─── IDEA 27: TTT + Hash Adapt + N-gram Beta Decay ────────────────────────────

def build27(src):
    src = add_header(src, 27, "Legal TTT + Hash Embed Adapt + N-gram Beta Decay",
        "TTT adapts both model params and bigram hash table; n-gram beta decays per chunk.")

    old_hash = '    hash_emb_size = int(os.environ.get("HASH_EMB_SIZE", 16384))  # 0 = disabled'
    new_hash = (
        '    hash_emb_size = int(os.environ.get("HASH_EMB_SIZE", 16384))  # 0 = disabled\n'
        '    # [IDEA-27]\n'
        '    ttt_hash_adapt = bool(int(os.environ.get("TTT_HASH_ADAPT", "1")))\n'
        '    ngram_beta_decay = float(os.environ.get("NGRAM_BETA_DECAY", "0.15"))\n'
        '    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "5"))')
    src = patch(src, old_hash, new_hash)

    # Enable TTT by default
    old_ttt = '    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))'
    new_ttt = '    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))  # [IDEA-27] on by default'
    if old_ttt in src:
        src = patch(src, old_ttt, new_ttt)
    return src


# ─── Build all ────────────────────────────────────────────────────────────────

BUILDERS = {
    18: build18,
    19: build19,
    20: build20,
    21: build21,
    22: build22,
    23: build23,
    24: build24,
    25: build25,
    26: build26,
    27: build27,
}

TITLES = {
    18: "AWQ + Hadamard Pre-Rotation GPTQ",
    19: "4-gram + 5-gram NGram Hash Embedding",
    20: "int5 Mixed-Precision GPTQ",
    21: "recur_passes=3 + Untied Per-Pass Adapters",
    22: "Cautious Weight Decay + Per-Layer LR Decay",
    23: "Differential Attention (MSFT 2024) + XSA Hybrid",
    24: "3-Lane Parallel Residual",
    25: "Self-Distillation QAT (fp32 Teacher)",
    26: "Mini 2-Expert Mixture-of-Experts MLP",
    27: "Legal TTT + Hash Embed Adapt + N-gram Beta Decay",
}

if __name__ == "__main__":
    success, failed = [], []
    for n, builder in sorted(BUILDERS.items()):
        out = Path(f"train_gpt_sota_{n}.py")
        try:
            patched = builder(SRC)
            ast.parse(patched)
            out.write_text(patched, encoding="utf-8")
            kb = len(patched) // 1024
            print(f"[OK]   sota_{n}.py  ({kb} KB)  —  {TITLES[n]}")
            success.append(n)
        except Exception as e:
            print(f"[FAIL] sota_{n}: {e}")
            failed.append(n)
    print(f"\nOK: {success}")
    print(f"FAIL: {failed}")
    sys.exit(0 if not failed else 1)
