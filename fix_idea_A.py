import re

with open('train_gpt_idea_A_PRNG_Context.py', 'r') as f:
    text = f.read()

text = text.replace(
    "torch.randn((self.context_values.size(0), x.size(-1))",
    "torch.randn((self.context_values.size(-1), x.size(-1))"
)

text = text.replace(
    "torch.nn.Parameter(torch.zeros(h.context_memory_dim, base_model.model_dim, device=device, dtype=torch.float32))",
    "torch.nn.Parameter(torch.zeros(base_model.model_dim, h.context_memory_dim, device=device, dtype=torch.float32))"
)

with open('train_gpt_idea_A_PRNG_Context.py', 'w') as f:
    f.write(text)

print("Idea A Fixed.")
