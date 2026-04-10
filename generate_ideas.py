import re

with open('/media/mlinh/DATA/projects/ML/lab-2_PG/parameter-golf/train_gpt_strategy_ABD.py', 'r') as f:
    base_src = f.read()

# =========================================================================
# IDEA A: PRNG-Contextual Memory Bank
# =========================================================================

idea_a_src = base_src.replace(
    "class Hyperparameters():",
    "class Hyperparameters():\n    # Idea A Context Memory\n    context_memory_enabled = bool(int(os.environ.get('CONTEXT_MEMORY_ENABLED', '1')))\n    context_memory_dim = int(os.environ.get('CONTEXT_MEMORY_DIM', 4096))\n    context_memory_seed = int(os.environ.get('CONTEXT_MEMORY_SEED', 1337))\n"
)

idea_a_injection = """
        # Idea A Contextual Memory Eval-time Injection
        if getattr(self, 'context_memory_active', False) and hasattr(self, 'context_values'):
            if not hasattr(self, '_context_proj_weight'):
                gen = torch.Generator(device=x.device).manual_seed(self.context_memory_seed)
                # Ensure reproducibility across runs/devices
                self._context_proj_weight = torch.randn((self.context_values.size(0), x.size(-1)), generator=gen, device=x.device, dtype=x.dtype)
                self._context_proj_weight /= math.sqrt(x.size(-1))
            
            context_keys = F.linear(x, self._context_proj_weight) # [B, T, V]
            sparse_act = F.relu(context_keys)
            context_out = F.linear(sparse_act, self.context_values.to(dtype=sparse_act.dtype))
            x = x + context_out
            
        x = self.final_norm(x)"""

idea_a_src = idea_a_src.replace("x = self.final_norm(x)", idea_a_injection)

idea_a_ttt_injection = """    if getattr(h, 'context_memory_enabled', False):
        base_model.context_memory_active = True
        base_model.context_values = torch.nn.Parameter(torch.zeros(h.context_memory_dim, base_model.model_dim, device=device, dtype=torch.float32))

    frozen_block_ids = set(range(min(h.ttt_freeze_blocks, len(base_model.blocks))))"""

idea_a_src = idea_a_src.replace(
    "    frozen_block_ids = set(range(min(h.ttt_freeze_blocks, len(base_model.blocks))))",
    idea_a_ttt_injection
)

with open('/media/mlinh/DATA/projects/ML/lab-2_PG/parameter-golf/train_gpt_idea_A_PRNG_Context.py', 'w') as f:
    f.write(idea_a_src)
print("Generated Idea A")

# =========================================================================
# IDEA B: Temporal SWA for TTT
# =========================================================================

idea_b_src = base_src.replace(
    "class Hyperparameters():",
    "class Hyperparameters():\n    # Idea B Temporal SWA\n    ttt_swa_decay = float(os.environ.get('TTT_SWA_DECAY', 0.95))\n"
).replace(
    "    for ci in range(num_chunks):",
    """    eval_ema_weights = {name: p.detach().clone().float() for name, p in base_model.named_parameters()}
    
    for ci in range(num_chunks):"""
).replace(
    "        # --- Phase 1: SCORE this chunk's windows (no_grad for TTT compat) ---",
    """        # --- Phase 1: SCORE this chunk's windows (no_grad for TTT compat) ---
        active_weights = {name: p.detach().clone() for name, p in base_model.named_parameters()}
        base_model.load_state_dict({name: p.to(dtype=active_weights[name].dtype) for name, p in eval_ema_weights.items()}, strict=True)
"""
).replace(
    "        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---",
    """        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        base_model.load_state_dict(active_weights, strict=True)"""
).replace(
    "                        optimizer.step()",
    """                        optimizer.step()
                        
        if epochs > 0:
            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    eval_ema_weights[name].mul_(h.ttt_swa_decay).add_(p.detach().float(), alpha=1.0 - h.ttt_swa_decay)"""
)

with open('/media/mlinh/DATA/projects/ML/lab-2_PG/parameter-golf/train_gpt_idea_B_TemporalSWA.py', 'w') as f:
    f.write(idea_b_src)
print("Generated Idea B")

# =========================================================================
# IDEA C: Recurrent Parallel Residual Blocks
# =========================================================================

idea_c_src = base_src

old_block_forward = """    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out"""

new_block_forward = """    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        
        # True Parallel Residual Block - attention and mlp share the same pre-norm input
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
        
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out \\
                     + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        return x_out"""

idea_c_src = idea_c_src.replace(old_block_forward, new_block_forward)

# Replace the messy forward_logits logic which used lane0 and lane1
# Find the start and end of forward_logits and swap it completely
old_forward_logits_pattern = r"    def forward_logits\(self, input_ids: Tensor\) -> Tensor:.*?        return self.logit_softcap \* torch.tanh\(logits_proj / self.logit_softcap\)"

new_forward_logits = """    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x

        virtual_layers = self._get_virtual_layers()
        num_virtual = len(virtual_layers)
        num_enc = num_virtual // 2
        num_dec = num_virtual - num_enc

        skips: list[Tensor] = []
        ve_cache: dict = {}

        # Encoder phase
        for vi in range(num_enc):
            phys_idx = virtual_layers[vi]
            ve = self._get_ve(phys_idx, input_ids, ve_cache)
            x = self.blocks[phys_idx](x, x0, v_embed=ve)
            skips.append(x)

        # Decoder phase with U-Net skip connections
        for vi in range(num_dec):
            phys_idx = virtual_layers[num_enc + vi]
            if skips and vi < self.num_skip_weights:
                scaled_skip = self.skip_weights[vi].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[vi].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip

            ve = self._get_ve(phys_idx, input_ids, ve_cache)
            x = self.blocks[phys_idx](x, x0, v_embed=ve)

        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)"""

idea_c_src = re.sub(old_forward_logits_pattern, new_forward_logits, idea_c_src, flags=re.DOTALL)

with open('/media/mlinh/DATA/projects/ML/lab-2_PG/parameter-golf/train_gpt_idea_C_ParResid_Block.py', 'w') as f:
    f.write(idea_c_src)
print("Generated Idea C")

