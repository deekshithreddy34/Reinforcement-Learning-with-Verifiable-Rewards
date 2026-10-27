

%%capture
# Install Unsloth + dependencies
# We patch TRL on disk to disable optional integrations (weave, mergekit, llm_blender, liger_kernel)
# This is the same trick from 10a — TRL eagerly imports things we don't need
!pip install -q unsloth datasets
!pip install -q --no-deps trl peft accelerate bitsandbytes

import os, glob, re

trl_dir = "/usr/local/lib/python3.12/dist-packages/trl"
problem_libs = ["weave", "mergekit", "llm_blender", "liger_kernel"]

def _patch_line(line):
    leading = line[:len(line) - len(line.lstrip())]
    s = line.strip()
    m = re.match(r"^from\s+(\w+)[\.\w]*\s+import\s+(.+)$", s)
    if m and m.group(1) in problem_libs:
        names = []
        for p in m.group(2).rstrip(")").split(","):
            p = p.strip()
            names.append(p.split(" as ")[1].strip() if " as " in p else p)
        stubs = "; ".join(f"{n} = None" for n in names if n)
        return f"{leading}{stubs}  # was: {s}\n"
    m = re.match(r"^import\s+(\w+)", s)
    if m and m.group(1) in problem_libs:
        return f"{leading}{m.group(1)} = None  # was: {s}\n"
    return line

if os.path.exists(trl_dir):
    for fp in glob.glob(os.path.join(trl_dir, "**/*.py"), recursive=True):
        with open(fp) as f:
            lines = f.readlines()
        new = [_patch_line(l) for l in lines]
        if new != lines:
            with open(fp, "w") as f:
                f.writelines(new)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
print("[OK] Environment ready")


import torch
import re
import random
import numpy as np

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


from unsloth import FastLanguageModel

max_seq_length = 1024  # Reasoning needs longer sequences
lora_rank = 64         # Higher rank than SFT/DPO — reasoning needs more capacity

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/SmolLM-135M-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False,  # vLLM doesn't support models this small
    max_lora_rank = lora_rank,
)

# Add LoRA — same target modules as Unsloth's official GRPO recipe
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

print(f"\n[OK] Model loaded: SmolLM-135M with LoRA rank {lora_rank}")
print(f"     Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")