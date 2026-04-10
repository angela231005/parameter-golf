import re

with open('train_gpt_idea_B_TemporalSWA.py', 'r') as f:
    text = f.read()

target = """    # Restore all params as trainable, put model in eval mode
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()"""

replacement = """    # Restore all params as trainable, put model in eval mode
    for p in base_model.parameters():
        p.requires_grad_(True)
    
    # [Idea B] Load the EMA weights into the model for the final evaluation
    if getattr(base_model, 'eval_ema_weights', None) is not None:
        base_model.load_state_dict({name: p.to(dtype=base_model.get_parameter(name).dtype) for name, p in base_model.eval_ema_weights.items()}, strict=True)
        
    base_model.eval()"""

text = text.replace(target, replacement)

# We also need to store eval_ema_weights in base_model so it's accessible above.
text = text.replace(
    "    eval_ema_weights = {name: p.detach().clone().float() for name, p in base_model.named_parameters()}",
    "    eval_ema_weights = {name: p.detach().clone().float() for name, p in base_model.named_parameters()}\n    base_model.eval_ema_weights = eval_ema_weights"
)

with open('train_gpt_idea_B_TemporalSWA.py', 'w') as f:
    f.write(text)

print("Idea B Fixed.")
