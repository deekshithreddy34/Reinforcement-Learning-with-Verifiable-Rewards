

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


from datasets import load_dataset

# The system prompt that tells the model what format to use
# This is straight from Unsloth's official GRPO recipe
SYSTEM_PROMPT = '''
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
'''

XML_COT_FORMAT = '''<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>'''

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return data

dataset = get_gsm8k_questions()
print(f"[OK] Loaded {len(dataset):,} training examples")
print(f"\nExample question: {dataset[0]['prompt'][1]['content'][:200]}")
print(f"Expected answer: {dataset[0]['answer']}")


# These are Will Brown's reward functions used in Unsloth's official GRPO notebook
# Source: github.com/willccbb/verifiers (with minor modifications)

def correctness_reward_func(prompts, completions, answer, **kwargs):
    '''+2.0 if the model's <answer> matches ground truth exactly.'''
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward_func(completions, **kwargs):
    '''+0.5 if the <answer> is an integer (sanity check — math answers are numbers).'''
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def strict_format_reward_func(completions, **kwargs):
    '''+0.5 for EXACT XML structure including newlines.'''
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs):
    '''+0.5 for XML structure but lenient on whitespace.'''
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def count_xml(text):
    '''Graduated XML tag counting — partial credit for each tag.'''
    count = 0.0
    if text.count("<reasoning>\n") == 1: count += 0.125
    if text.count("\n</reasoning>\n") == 1: count += 0.125
    if text.count("\n<answer>\n") == 1: count += 0.125
    # Penalize text after </answer>
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>")[-1]) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs):
    '''Graduated reward — partial credit for each XML tag present.'''
    contents = [c[0]["content"] for c in completions]
    return [count_xml(c) for c in contents]

print("[OK] 5 reward functions defined")
print("     Max possible reward per response: +4.0")

# Three test responses, increasing in quality

def fake_completion(text):
    '''Mimic the structure GRPO trainer passes — list of [{role, content}].'''
    return [[{"role": "assistant", "content": text}]]

# 1. Garbage response — no format, no structure
test1 = fake_completion("the answer is probably 42 because reasons")

# 2. Has format but wrong answer
test2 = fake_completion('''<reasoning>
let me think
</reasoning>
<answer>
42
</answer>''')

# 3. Perfect — format AND right answer
test3 = fake_completion('''<reasoning>
18 multiplied by 7 is 126
</reasoning>
<answer>
126
</answer>''')

ground_truth = ["126"]

for i, c in enumerate([test1, test2, test3], 1):
    print(f"\n=== Test {i} ===")
    print(f"Response: {c[0][0]['content'][:80]}...")
    print(f"  correctness:    {correctness_reward_func(None, c, ground_truth)[0]:+.2f}")
    print(f"  int_reward:     {int_reward_func(c)[0]:+.2f}")
    print(f"  strict_format:  {strict_format_reward_func(c)[0]:+.2f}")
    print(f"  soft_format:    {soft_format_reward_func(c)[0]:+.2f}")
    print(f"  xmlcount:       {xmlcount_reward_func(c)[0]:+.4f}")
    total = (correctness_reward_func(None, c, ground_truth)[0]
             + int_reward_func(c)[0]
             + strict_format_reward_func(c)[0]
             + soft_format_reward_func(c)[0]
             + xmlcount_reward_func(c)[0])
    print(f"  TOTAL:          {total:+.4f}")


# Generate a baseline response BEFORE GRPO training
FastLanguageModel.for_inference(model)

# Take a question from the dataset
sample_q = dataset[0]
prompt = tokenizer.apply_chat_template(
    sample_q["prompt"],
    tokenize=False,
    add_generation_prompt=True,
)

print("=== QUESTION ===")
print(sample_q["prompt"][1]["content"])
print(f"\n=== EXPECTED ANSWER ===")
print(sample_q["answer"])

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n=== BASELINE MODEL OUTPUT (before GRPO) ===")
print(response)
print(f"\n=== AUTOMATED REWARD SCORE ===")
fake = fake_completion(response)
total = (correctness_reward_func(None, fake, [sample_q["answer"]])[0]
         + int_reward_func(fake)[0]
         + strict_format_reward_func(fake)[0]
         + soft_format_reward_func(fake)[0]
         + xmlcount_reward_func(fake)[0])
print(f"Total reward: {total:+.4f} (out of max +4.0)")


from unsloth import PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer

# Critical hyperparameters for GRPO
training_args = GRPOConfig(
    output_dir = "outputs",

    # --- Learning rate (same as DPO range) ---
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",

    # --- The "G" in GRPO ---
    num_generations = 4,         # Reduced from default 6 for T4 memory
    # Each step generates 4 different responses to the SAME prompt
    # The advantage is computed RELATIVE to these 4 responses

    # --- Batch sizes ---
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,

    # --- Sequence lengths ---
    max_prompt_length = 256,
    max_completion_length = 512,

    # --- Training duration ---
    max_steps = 100,             # Demo only — production needs 300+
    save_steps = 50,
    logging_steps = 1,

    # --- Other ---
    max_grad_norm = 0.1,
    report_to = "none",
    seed = SEED,

    # --- Use TRL's reference (no vLLM for tiny models) ---
    use_vllm = False,
)

print("[OK] GRPOConfig ready")
print(f"     Group size (num_generations): {training_args.num_generations}")
print(f"     Max steps: {training_args.max_steps}")
print(f"     Learning rate: {training_args.learning_rate}")

# Switch back to training mode and train
FastLanguageModel.for_training(model)

trainer_stats = trainer.train()

print("\n[OK] GRPO training complete")
print(f"Final loss: {trainer_stats.training_loss:.4f}")


# Generate after-training output on the same question
FastLanguageModel.for_inference(model)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

response_after = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("=== QUESTION ===")
print(sample_q["prompt"][1]["content"])
print(f"\n=== EXPECTED ANSWER ===")
print(sample_q["answer"])

print(f"\n=== AFTER GRPO TRAINING ===")
print(response_after)
print(f"\n=== REWARD ===")
fake = fake_completion(response_after)
total = (correctness_reward_func(None, fake, [sample_q["answer"]])[0]
         + int_reward_func(fake)[0]
         + strict_format_reward_func(fake)[0]
         + soft_format_reward_func(fake)[0]
         + xmlcount_reward_func(fake)[0])
print(f"Total: {total:+.4f}")


import matplotlib.pyplot as plt

# Pull metrics from the trainer's log history
log_history = trainer.state.log_history

steps, rewards, reward_stds, completion_lengths, kls = [], [], [], [], []
for entry in log_history:
    if "reward" in entry:
        steps.append(entry.get("step", 0))
        rewards.append(entry.get("reward", 0))
        reward_stds.append(entry.get("reward_std", 0))
        completion_lengths.append(entry.get("completion_length", 0))
        kls.append(entry.get("kl", 0))

if not steps:
    print("[WARN] No reward logs found")
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0,0].plot(steps, rewards, color='#3498db')
    axes[0,0].set_title("Total Reward over training")
    axes[0,0].set_xlabel("Step"); axes[0,0].set_ylabel("Reward")
    axes[0,0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(steps, reward_stds, color='#e74c3c')
    axes[0,1].set_title("Reward Std (group diversity)")
    axes[0,1].set_xlabel("Step"); axes[0,1].set_ylabel("Std")
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(steps, completion_lengths, color='#2ecc71')
    axes[1,0].set_title("Completion Length")
    axes[1,0].set_xlabel("Step"); axes[1,0].set_ylabel("Tokens")
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(steps, kls, color='#9b59b6')
    axes[1,1].set_title("KL from Reference")
    axes[1,1].set_xlabel("Step"); axes[1,1].set_ylabel("KL")
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/content/grpo_metrics.png", dpi=80)
    plt.show()
    print(f"\n[OK] Plotted {len(steps)} steps")


# Sample 20 prompts and break down the reward
N_SAMPLES = 20

FastLanguageModel.for_inference(model)
breakdown = {
    "correctness": [], "int": [], "strict_format": [],
    "soft_format": [], "xmlcount": [],
}

import random
test_indices = random.sample(range(len(dataset)), N_SAMPLES)

print("Sampling 20 random training prompts post-GRPO...\n")
for i, idx in enumerate(test_indices):
    sample = dataset[idx]
    p = tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
    inp = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=300, do_sample=True,
            temperature=0.8, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    fake = fake_completion(resp)

    breakdown["correctness"].append(correctness_reward_func(None, fake, [sample["answer"]])[0])
    breakdown["int"].append(int_reward_func(fake)[0])
    breakdown["strict_format"].append(strict_format_reward_func(fake)[0])
    breakdown["soft_format"].append(soft_format_reward_func(fake)[0])
    breakdown["xmlcount"].append(xmlcount_reward_func(fake)[0])

    print(f"  [{i+1:2d}/{N_SAMPLES}] correct={breakdown['correctness'][-1]:.1f}  "
          f"format={breakdown['strict_format'][-1]:.1f}  xml={breakdown['xmlcount'][-1]:.3f}")

print(f"\n=== AVERAGE REWARD BY FUNCTION (across {N_SAMPLES} samples) ===")
for k, v in breakdown.items():
    avg = sum(v) / len(v)
    pct_nonzero = 100 * sum(1 for x in v if x > 0) / len(v)
    print(f"  {k:<14}: avg={avg:+.4f}  fired_in={pct_nonzero:5.1f}% of samples")


# Diagnostic summary

avg_correct = sum(breakdown["correctness"]) / len(breakdown["correctness"])
avg_format = sum(breakdown["strict_format"]) / len(breakdown["strict_format"])
avg_xml = sum(breakdown["xmlcount"]) / len(breakdown["xmlcount"])

# Reward std diagnostic
import numpy as np
final_std = np.mean(reward_stds[-10:]) if reward_stds else 0
initial_std = np.mean(reward_stds[:10]) if len(reward_stds) >= 10 else 0

print("=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
print(f"\n1. Did reasoning emerge?")
print(f"   correctness_reward avg: {avg_correct:.4f}")
if avg_correct > 0.5:
    print("   --> YES, model is producing some correct answers")
elif avg_correct > 0.1:
    print("   --> PARTIAL, occasional correct answers")
else:
    print("   --> NO, model essentially never produces correct answers")

print(f"\n2. Is the model exploring different responses?")
print(f"   reward_std initial: {initial_std:.4f}")
print(f"   reward_std final:   {final_std:.4f}")
if final_std < 0.01:
    print("   --> COLLAPSED: all generations identical, no learning signal")
elif final_std > initial_std * 1.5:
    print("   --> HEALTHY: model exploring different responses")
else:
    print("   --> WEAK: limited diversity in generations")

print(f"\n3. Did the model learn the format?")
print(f"   strict_format_reward avg: {avg_format:.4f}")
print(f"   xmlcount_reward avg:      {avg_xml:.4f}")
if avg_format > 0.3:
    print("   --> YES, model produces correct XML structure")
elif avg_xml > 0.1:
    print("   --> PARTIAL, model produces some XML tags")
else:
    print("   --> NO, model doesn't reliably produce structure")

print(f"\n" + "=" * 60)
print("THE LESSON")
print("=" * 60)
print('''
SmolLM-135M is below the model-size floor for GRPO.

Reason: GRPO REINFORCES existing behaviors. It does not
TEACH new ones from scratch.

For reasoning to emerge, the base model must occasionally:
  - Produce something that looks like reasoning
  - Produce structured outputs (XML tags)
  - Get math problems right by chance

A 135M-parameter model rarely does any of these.
A 1.5B model occasionally does ALL of them.

Once the base model occasionally produces good outputs, GRPO
amplifies that frequency. Without that seed behavior, GRPO
just bounces randomly with no signal to lock onto.

This is THE practical lesson of RL for LLMs:
  Algorithm capacity is bounded by base-model capacity.
  RL polishes. RL amplifies. RL does not teach.
''')