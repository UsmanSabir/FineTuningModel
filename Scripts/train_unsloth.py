"""
═══════════════════════════════════════════════════════════════════
  Restaurant AI Agent — Fine-Tuning & Validation with Unsloth
═══════════════════════════════════════════════════════════════════

Install dependencies first:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers trl peft accelerate bitsandbytes
  pip install datasets transformers torch

Usage:
  # Train with default settings (llama-3.2-3b on your dataset)
  python train_unsloth.py

  # Custom model and dataset
  python train_unsloth.py --model unsloth/Qwen2.5-7B-Instruct --dataset my_dataset.jsonl

  # Quick test run (5 steps only)
  python train_unsloth.py --test-run

  # Export to GGUF for Ollama after training
  python train_unsloth.py --export-gguf

  # Skip training, only run validation on a saved model
  python train_unsloth.py --skip-train --model-path ./bella_model
"""

import os
import json
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    # (model_id, context_length, description)
    "llama-3.2-3b":    ("unsloth/Llama-3.2-3B-Instruct",      8192,  "Fast, lightweight — good for testing"),
    "llama-3.2-1b":    ("unsloth/Llama-3.2-1B-Instruct",      8192,  "Smallest / fastest"),
    "llama-3.1-8b":    ("unsloth/Meta-Llama-3.1-8B-Instruct", 8192,  "Balanced quality/speed"),
    "qwen2.5-7b":      ("unsloth/Qwen2.5-7B-Instruct",        8192,  "Strong instruction following"),
    "qwen2.5-3b":      ("unsloth/Qwen2.5-3B-Instruct",        8192,  "Lightweight Qwen"),
    "mistral-7b":      ("unsloth/mistral-7b-instruct-v0.3",   4096,  "Classic strong performer"),
    "phi-3.5-mini":    ("unsloth/Phi-3.5-mini-instruct",      4096,  "Microsoft Phi — very efficient"),
    "gemma-2-9b":      ("unsloth/gemma-2-9b-it",              4096,  "Google Gemma 2"),
}

DEFAULT_MODEL = "llama-3.2-3b"

TRAIN_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler": "cosine",
    "weight_decay": 0.01,
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,  # effective batch = 8
    "max_seq_length": 4096,
    "seed": 42,
    "fp16": True,    # set False on newer GPUs that support bf16
    "bf16": False,   # set True for A100/H100
}

# ─────────────────────────────────────────────────────────────
# TOOL + SYSTEM PROMPT (keep in sync with dataset)
# ─────────────────────────────────────────────────────────────

TOOLS_JSON = json.dumps([
    {"name": "get_menu", "description": "Fetch the current menu.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "enum": ["appetizers", "mains", "desserts", "drinks", "all"]}}, "required": ["category"]}},
    {"name": "create_order", "description": "Place a new order.", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string"}, "items": {"type": "array"}, "order_type": {"type": "string", "enum": ["dine_in", "takeout", "delivery"]}}, "required": ["customer_name", "items", "order_type"]}},
    {"name": "get_order_status", "description": "Check order status.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "cancel_order", "description": "Cancel an order.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}},
    {"name": "process_refund", "description": "Initiate a refund.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}, "refund_amount": {"type": "number"}}, "required": ["order_id", "reason", "refund_amount"]}},
    {"name": "make_reservation", "description": "Reserve a table.", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string"}, "phone": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "party_size": {"type": "integer"}}, "required": ["customer_name", "phone", "date", "time", "party_size"]}},
], indent=2)

SYSTEM_PROMPT = f"""You are Bella, a friendly and efficient AI agent for Bella Italia Restaurant. You help customers with menu browsing, orders, reservations, and complaints.

Available tools:
{TOOLS_JSON}

Tool call format:
<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>

Use <think>...</think> for step-by-step reasoning before complex responses.
Be warm, concise, and solution-focused."""


# ─────────────────────────────────────────────────────────────
# DATASET LOADING & FORMATTING
# ─────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str, val_split: float = 0.15) -> tuple:
    """Load JSONL dataset and split into train/val sets."""
    print(f"\n📂 Loading dataset from: {path}")

    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Remove _meta keys (internal use only)
    for item in raw:
        item.pop("_meta", None)

    random.shuffle(raw)
    val_size = max(1, int(len(raw) * val_split))
    train_data = raw[val_size:]
    val_data = raw[:val_size]

    print(f"  Total examples  : {len(raw)}")
    print(f"  Train examples  : {len(train_data)}")
    print(f"  Val examples    : {len(val_data)}")
    return train_data, val_data


def format_conversation(example: dict, tokenizer) -> dict:
    """
    Convert a messages list into a tokenizer-formatted training string.
    Handles system/user/assistant/tool roles.
    """
    messages = example.get("messages", [])
    # Remap 'tool' role to a format the tokenizer can handle
    formatted = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "tool":
            # Wrap tool results as a user message with clear prefix
            formatted.append({"role": "user", "content": f"[TOOL RESULT]\n{content}"})
        else:
            formatted.append({"role": role, "content": content})

    try:
        text = tokenizer.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: manual formatting
        text = ""
        for m in formatted:
            text += f"<|{m['role']}|>\n{m['content']}\n"

    return {"text": text}


def prepare_hf_dataset(data: list, tokenizer, max_seq_length: int):
    """Convert list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset

    formatted = [format_conversation(ex, tokenizer) for ex in data]
    ds = Dataset.from_list(formatted)

    # Filter out examples that are too long
    def tokenize_and_filter(batch):
        tokens = tokenizer(batch["text"], truncation=False)
        lengths = [len(ids) for ids in tokens["input_ids"]]
        keep = [l <= max_seq_length for l in lengths]
        return {"text": [t for t, k in zip(batch["text"], keep) if k], "length": [l for l, k in zip(lengths, keep) if k]}

    before = len(ds)
    ds = ds.map(tokenize_and_filter, batched=True, remove_columns=["text"],
                batch_size=32).rename_column("text", "text") if False else ds

    # Simple filter
    tokenized = tokenizer(ds["text"], truncation=False)
    keep_idx = [i for i, ids in enumerate(tokenized["input_ids"]) if len(ids) <= max_seq_length]
    if len(keep_idx) < len(ds):
        print(f"  ⚠️  Filtered {len(ds) - len(keep_idx)} examples exceeding {max_seq_length} tokens")
        ds = ds.select(keep_idx)

    return ds


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, max_seq_length: int, cfg: dict):
    """Load model + tokenizer with Unsloth 4-bit quantization and LoRA."""
    from unsloth import FastLanguageModel

    print(f"\n🤖 Loading model: {model_id}")
    print(f"   Max seq length : {max_seq_length}")
    print(f"   LoRA r         : {cfg['lora_r']} | alpha: {cfg['lora_alpha']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,           # auto-detect
        load_in_4bit=True,    # 4-bit quantization — reduces VRAM ~75%
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves VRAM
        random_state=cfg["seed"],
        use_rslora=True,    # Rank-Stabilized LoRA — usually better
        loftq_config=None,
    )

    # Print trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, val_ds, output_dir: str, cfg: dict, test_run: bool = False):
    """Run SFT training with TRL's SFTTrainer."""
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq

    max_steps = 10 if test_run else -1
    num_epochs = 1 if test_run else cfg["num_epochs"]

    print(f"\n🏋️  Starting training...")
    print(f"   Output dir  : {output_dir}")
    print(f"   Epochs      : {num_epochs}")
    print(f"   Max steps   : {'10 (test run)' if test_run else 'unlimited'}")
    print(f"   Batch size  : {cfg['per_device_train_batch_size']} × {cfg['gradient_accumulation_steps']} accum = {cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']} effective\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        optim="adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg["seed"],
        report_to="none",           # set to "wandb" if you use Weights & Biases
        dataloader_num_workers=0,
        group_by_length=True,       # speeds up training by grouping similar-length examples
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=2,
        packing=True,               # pack multiple short examples per context window
        args=training_args,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    print(f"\n✅ Training complete!")
    print(f"   Time elapsed : {elapsed/60:.1f} minutes")
    print(f"   Final loss   : {result.training_loss:.4f}")

    return trainer, result


# ─────────────────────────────────────────────────────────────
# VALIDATION & EVALUATION
# ─────────────────────────────────────────────────────────────

VALIDATION_PROMPTS = [
    {
        "name": "Menu query",
        "user": "What vegetarian options do you have?",
        "expect_tool": "get_menu",
        "expect_keywords": ["menu", "vegetarian", "risotto", "pizza"],
    },
    {
        "name": "Order placement",
        "user": "I'd like a Margherita Pizza for delivery. Name is Alex, address is 10 Main St.",
        "expect_tool": "create_order",
        "expect_keywords": ["order", "confirmed", "delivery", "pizza"],
    },
    {
        "name": "Order status",
        "user": "Where is my order? Order number ORD-1234",
        "expect_tool": "get_order_status",
        "expect_keywords": ["order", "ORD-1234"],
    },
    {
        "name": "Refund request",
        "user": "My food was terrible. I want my money back. Order ORD-5678.",
        "expect_tool": "process_refund",
        "expect_keywords": ["refund", "sorry", "apol"],
    },
    {
        "name": "Reservation",
        "user": "Can I book a table for 2 this Friday at 8pm? Name is John, phone 555-1234.",
        "expect_tool": "make_reservation",
        "expect_keywords": ["reservation", "table", "confirmed"],
    },
    {
        "name": "Out of scope",
        "user": "What's the weather like today?",
        "expect_tool": None,
        "expect_keywords": ["restaurant", "menu", "order", "help"],
    },
    {
        "name": "Thinking trace",
        "user": "I have a shellfish allergy and need gluten-free food. What do you recommend?",
        "expect_tool": "get_menu",
        "expect_keywords": ["allergy", "gluten"],
    },
    {
        "name": "Angry customer",
        "user": "This is absolutely unacceptable! My order has been wrong THREE times!",
        "expect_tool": None,
        "expect_keywords": ["sorry", "apol", "understand"],
    },
]


def run_inference(model, tokenizer, user_message: str, max_new_tokens: int = 512) -> str:
    """Run inference on the fine-tuned model."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def validate_model(model, tokenizer, prompts: list = None) -> dict:
    """Run validation prompts and score the model."""
    if prompts is None:
        prompts = VALIDATION_PROMPTS

    print("\n" + "═" * 60)
    print("  🔍 VALIDATION RESULTS")
    print("═" * 60)

    results = []

    for i, vp in enumerate(prompts, 1):
        print(f"\n  [{i}/{len(prompts)}] {vp['name']}")
        print(f"  User: {vp['user'][:80]}...")

        response = run_inference(model, tokenizer, vp["user"])
        print(f"  Bot:  {response[:200]}{'...' if len(response) > 200 else ''}")

        # Score
        scores = {}
        response_lower = response.lower()

        # Check for tool calls
        has_tool_call = "<tool_call>" in response
        if vp["expect_tool"]:
            tool_hit = vp["expect_tool"] in response
            scores["tool_call"] = 1.0 if tool_hit else 0.0
            status = "✅" if tool_hit else "❌"
            print(f"  Tool call [{vp['expect_tool']}]: {status}")
        else:
            if has_tool_call:
                scores["tool_call"] = 0.5  # tool called when not expected
                print(f"  No tool expected — tool was called anyway ⚠️")
            else:
                scores["tool_call"] = 1.0
                print(f"  No tool expected: ✅")

        # Check for keywords
        keyword_hits = [kw for kw in vp["expect_keywords"] if kw.lower() in response_lower]
        scores["keywords"] = len(keyword_hits) / len(vp["expect_keywords"]) if vp["expect_keywords"] else 1.0
        print(f"  Keywords [{len(keyword_hits)}/{len(vp['expect_keywords'])}]: {', '.join(keyword_hits) or 'none'}")

        # Check for thinking traces
        has_think = "<think>" in response
        scores["has_think"] = 1.0 if has_think else 0.0
        print(f"  Think trace: {'✅' if has_think else '—'}")

        # Check response quality (not too short, not too long)
        response_len = len(response.split())
        if response_len < 10:
            scores["length"] = 0.0
        elif response_len > 500:
            scores["length"] = 0.7
        else:
            scores["length"] = 1.0

        overall = sum(scores.values()) / len(scores)
        scores["overall"] = overall
        print(f"  Score: {overall:.0%}")

        results.append({"prompt": vp["name"], "scores": scores, "response": response})

    # Summary
    avg_score = sum(r["scores"]["overall"] for r in results) / len(results)
    print("\n" + "═" * 60)
    print(f"  📈 AVERAGE SCORE: {avg_score:.0%}")

    if avg_score >= 0.85:
        print("  Rating: 🌟 Excellent — model is performing well!")
    elif avg_score >= 0.70:
        print("  Rating: ✅ Good — minor issues to address")
    elif avg_score >= 0.50:
        print("  Rating: ⚠️  Fair — consider more training data")
    else:
        print("  Rating: ❌ Poor — check dataset quality and training config")

    print("═" * 60)

    return {"average_score": avg_score, "results": results}


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_model(model, tokenizer, output_dir: str, export_gguf: bool = False):
    """Save the fine-tuned model in various formats."""
    from unsloth import FastLanguageModel

    print(f"\n💾 Saving model to: {output_dir}")

    # Save LoRA adapters (small, ~100MB)
    lora_path = f"{output_dir}/lora_adapters"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"  ✅ LoRA adapters saved: {lora_path}")

    # Save merged 16-bit model (full model, larger)
    merged_path = f"{output_dir}/merged_16bit"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"  ✅ Merged 16-bit model saved: {merged_path}")

    if export_gguf:
        # Export GGUF for use with Ollama / llama.cpp
        gguf_path = f"{output_dir}/gguf"
        print(f"  🔄 Exporting GGUF (Q4_K_M quantization)...")
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"  ✅ GGUF saved: {gguf_path}")
        print(f"\n  To use with Ollama:")
        print(f"    ollama create bella-italia -f {gguf_path}/Modelfile")
        print(f"    ollama run bella-italia")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune restaurant AI agent with Unsloth")

    # Model options
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(SUPPORTED_MODELS.keys()) + ["custom"],
                        help=f"Model preset to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-id", default=None,
                        help="Custom Hugging Face model ID (use with --model custom)")
    parser.add_argument("--model-path", default=None,
                        help="Path to a locally saved model (skips download)")

    # Dataset options
    parser.add_argument("--dataset", default="restaurant_finetune_dataset.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data to use for validation (default: 0.15)")

    # Training options
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"],
                        help=f"Training epochs (default: {TRAIN_CONFIG['num_epochs']})")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"],
                        help=f"Learning rate (default: {TRAIN_CONFIG['learning_rate']})")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"],
                        help=f"Per-device batch size (default: {TRAIN_CONFIG['per_device_train_batch_size']})")
    parser.add_argument("--lora-r", type=int, default=TRAIN_CONFIG["lora_r"],
                        help=f"LoRA rank (default: {TRAIN_CONFIG['lora_r']})")
    parser.add_argument("--max-seq-len", type=int, default=TRAIN_CONFIG["max_seq_length"],
                        help=f"Max sequence length (default: {TRAIN_CONFIG['max_seq_length']})")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 instead of FP16 (for A100/H100 GPUs)")

    # Run modes
    parser.add_argument("--test-run", action="store_true",
                        help="Run only 10 training steps (quick sanity check)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training — only run validation on an existing model")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation after training")
    parser.add_argument("--export-gguf", action="store_true",
                        help="Export GGUF file for Ollama after training")

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not set)")

    args = parser.parse_args()

    # ── Resolve model ──
    if args.model == "custom" and args.model_id:
        model_id = args.model_id
        max_seq_length = args.max_seq_len
    elif args.model_path:
        model_id = args.model_path
        max_seq_length = args.max_seq_len
    else:
        model_id, max_seq_length, desc = SUPPORTED_MODELS[args.model]
        print(f"  Model: {args.model} — {desc}")

    if args.max_seq_len != TRAIN_CONFIG["max_seq_length"]:
        max_seq_length = args.max_seq_len

    # ── Apply arg overrides to config ──
    cfg = TRAIN_CONFIG.copy()
    cfg["num_epochs"] = args.epochs
    cfg["learning_rate"] = args.lr
    cfg["per_device_train_batch_size"] = args.batch_size
    cfg["lora_r"] = args.lora_r
    cfg["lora_alpha"] = args.lora_r * 2
    cfg["max_seq_length"] = max_seq_length
    if args.bf16:
        cfg["bf16"] = True
        cfg["fp16"] = False

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./bella_model_{args.model}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ── Banner ──
    print("═" * 60)
    print("  🍝 Bella Italia — Unsloth Fine-Tuning")
    print("═" * 60)
    print(f"  Model        : {model_id}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Epochs       : {cfg['num_epochs']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  LoRA r       : {cfg['lora_r']} (alpha: {cfg['lora_alpha']})")
    print(f"  Seq length   : {max_seq_length}")
    print(f"  Precision    : {'BF16' if cfg['bf16'] else 'FP16'}")
    print(f"  Test run     : {args.test_run}")
    print("═" * 60)

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(model_id, max_seq_length, cfg)

    if not args.skip_train:
        # ── Load & prepare dataset ──
        if not Path(args.dataset).exists():
            print(f"\n❌ Dataset not found: {args.dataset}")
            print("   Run generate_dataset.py or generate_dataset_ai.py first.")
            return

        train_data, val_data = load_dataset_from_jsonl(args.dataset, args.val_split)

        print("\n📝 Formatting dataset...")
        train_ds = prepare_hf_dataset(train_data, tokenizer, max_seq_length)
        val_ds = prepare_hf_dataset(val_data, tokenizer, max_seq_length)

        print(f"  Train dataset size : {len(train_ds)}")
        print(f"  Val dataset size   : {len(val_ds)}")

        # ── Train ──
        trainer, result = train(model, tokenizer, train_ds, val_ds, output_dir, cfg, args.test_run)

        # ── Save training config ──
        config_path = f"{output_dir}/training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_id": model_id,
                "dataset": args.dataset,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "config": cfg,
                "final_loss": result.training_loss,
                "timestamp": timestamp,
            }, f, indent=2)
        print(f"\n  Config saved: {config_path}")

    # ── Validate ──
    if not args.skip_validate:
        print("\n⚙️  Switching model to inference mode for validation...")
        val_results = validate_model(model, tokenizer)

        # Save validation results
        val_path = f"{output_dir}/validation_results.json"
        with open(val_path, "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        print(f"\n  Validation results saved: {val_path}")

    # ── Export ──
    if not args.skip_train:
        export_model(model, tokenizer, output_dir, args.export_gguf)

    print(f"\n🎉 All done! Model and results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
