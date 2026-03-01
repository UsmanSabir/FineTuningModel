"""
═══════════════════════════════════════════════════════════════════
  Restaurant AI Agent — Fine-Tuning v2 (Native tool_calls format)
  Handles: tool_calls[], tool_call_id, reasoning field, tools key
═══════════════════════════════════════════════════════════════════

Usage:
  python train_unsloth_v2.py                          # default
  python train_unsloth_v2.py --model qwen2.5-7b       # different model
  python train_unsloth_v2.py --test-run               # 10-step sanity check
  python train_unsloth_v2.py --export-gguf            # export for Ollama
  python train_unsloth_v2.py --skip-train --validate  # validate only
"""

import os, json, time, argparse, random
from pathlib import Path
from datetime import datetime

# validate_example lives in import_dataset.py — single source of truth
try:
    from import_dataset import validate_example
except ImportError:
    raise ImportError(
        "import_dataset.py not found in the same directory.\n"
        "Place it alongside this script — validate_example is shared across all scripts."
    )

# ─────────────────────────────────────────────────────────────
# SUPPORTED MODELS
# ─────────────────────────────────────────────────────────────

MODELS = {
    "llama-3.2-3b": ("unsloth/Llama-3.2-3B-Instruct",       8192),
    "llama-3.2-1b": ("unsloth/Llama-3.2-1B-Instruct",       8192),
    "llama-3.1-8b": ("unsloth/Meta-Llama-3.1-8B-Instruct",  8192),
    "qwen2.5-7b":   ("unsloth/Qwen2.5-7B-Instruct",         8192),
    "qwen2.5-3b":   ("unsloth/Qwen2.5-3B-Instruct",         8192),
    "mistral-7b":   ("unsloth/mistral-7b-instruct-v0.3",    4096),
    "phi-3.5-mini": ("unsloth/Phi-3.5-mini-instruct",       4096),
}

TRAIN_CFG = {
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "learning_rate": 2e-4, "warmup_ratio": 0.05, "lr_scheduler": "cosine",
    "weight_decay": 0.01, "num_epochs": 3,
    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
    "max_seq_length": 4096, "seed": 42, "fp16": True, "bf16": False,
}

# ─────────────────────────────────────────────────────────────
# FORMAT CONVERTER
# Native tool_calls → text the tokenizer can train on
# ─────────────────────────────────────────────────────────────

def serialize_tool_calls(tool_calls: list) -> str:
    """Convert tool_calls array to a compact, learnable string."""
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except Exception:
            args = {}
        parts.append(f'<tool_call>\n{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}\n</tool_call>')
    return "\n".join(parts)


def convert_message(m: dict) -> dict:
    """
    Convert a single message from native tool_calls format to a 
    chat-template-compatible format.

    Mapping:
      assistant + tool_calls   → role=assistant, content = reasoning block + tool_call tags
      role=tool                → role=user (tool result wrapper)
      assistant + content      → role=assistant, content as-is (with reasoning prepended)
    """
    role = m["role"]
    content = m.get("content", "") or ""
    reasoning = m.get("reasoning", "")
    tool_calls = m.get("tool_calls", [])

    if role == "assistant" and tool_calls:
        # Tool-calling turn: reasoning + serialized tool calls
        parts = []
        if reasoning:
            parts.append(f"<think>\n{reasoning}\n</think>")
        parts.append(serialize_tool_calls(tool_calls))
        return {"role": "assistant", "content": "\n\n".join(parts)}

    elif role == "tool":
        # Tool result: wrap as a special user message
        name = m.get("name", "tool")
        call_id = m.get("tool_call_id", "")
        return {
            "role": "user",
            "content": f"[TOOL_RESULT name={name} id={call_id}]\n{content}"
        }

    elif role == "assistant":
        # Regular reply: optionally prepend reasoning
        parts = []
        if reasoning:
            parts.append(f"<think>\n{reasoning}\n</think>")
        parts.append(content)
        return {"role": "assistant", "content": "\n\n".join(filter(None, parts))}

    else:
        # system / user — pass through unchanged
        return {"role": role, "content": content}


def format_example(example: dict, tokenizer) -> dict:
    """Convert a full training example to a tokenizable text string."""
    messages = example.get("messages", [])

    # Convert all messages
    converted = [convert_message(m) for m in messages]

    # Merge consecutive same-role messages (some tokenizers dislike back-to-back same role)
    merged = []
    for m in converted:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})

    try:
        text = tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback manual format
        text = ""
        for m in merged:
            text += f"### {m['role'].upper()}\n{m['content']}\n\n"

    return {"text": text}


# ─────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_dataset(path: str, val_split: float = 0.15, tokenizer=None, max_seq_length: int = 4096):
    from datasets import Dataset

    print(f"\n📂 Loading dataset: {path}")
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    for item in raw:
        item.pop("_meta", None)

    random.shuffle(raw)
    val_n = max(1, int(len(raw) * val_split))
    train_raw, val_raw = raw[val_n:], raw[:val_n]

    print(f"  Total : {len(raw)} | Train : {len(train_raw)} | Val : {len(val_raw)}")

    def to_text(batch_items):
        return [format_example(ex, tokenizer)["text"] for ex in batch_items]

    train_texts = to_text(train_raw)
    val_texts   = to_text(val_raw)

    # Filter by length
    def filter_length(texts):
        kept = []
        dropped = 0
        for t in texts:
            ids = tokenizer(t, truncation=False)["input_ids"]
            if len(ids) <= max_seq_length:
                kept.append(t)
            else:
                dropped += 1
        if dropped:
            print(f"  ⚠️  Dropped {dropped} examples exceeding {max_seq_length} tokens")
        return kept

    train_texts = filter_length(train_texts)
    val_texts   = filter_length(val_texts)

    print(f"  After filter — Train : {len(train_texts)} | Val : {len(val_texts)}")
    return Dataset.from_list([{"text": t} for t in train_texts]), \
           Dataset.from_list([{"text": t} for t in val_texts])


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def load_model(model_id, max_seq_length, cfg):
    from unsloth import FastLanguageModel

    print(f"\n🤖 Loading: {model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"], use_rslora=True,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, val_ds, output_dir, cfg, test_run=False):
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n🏋️  Training — output: {output_dir}")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if test_run else cfg["num_epochs"],
        max_steps=10 if test_run else -1,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"], bf16=cfg["bf16"],
        optim="adamw_8bit",
        logging_steps=5,
        save_strategy="epoch", eval_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg["seed"], report_to="none",
        group_by_length=True,
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=2, packing=True, args=args,
    )
    t0 = time.time()
    result = trainer.train()
    print(f"\n✅ Done in {(time.time()-t0)/60:.1f} min | Loss: {result.training_loss:.4f}")
    return trainer, result


# ─────────────────────────────────────────────────────────────
# VALIDATION
# Validates on native tool_calls format — checks model outputs
# proper tool calls matching the log patterns
# ─────────────────────────────────────────────────────────────

VALIDATION_CASES = [
    {
        "name": "Startup → GetCustomerDetails",
        "messages": [
            {"role": "system", "content": "You are a restaurant ordering assistant. On every new conversation, call GetCustomerDetails first."},
            {"role": "user", "content": "Hi, I want to order food"}
        ],
        "expect_tool": "GetCustomerDetails",
        "expect_keywords": [],
    },
    {
        "name": "Closed restaurant response",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Restaurant status: Closed. Opening at 8am tomorrow."},
            {"role": "user", "content": "kya aap open hain?"},
        ],
        "expect_tool": None,
        "expect_keywords": ["closed", "open", "8"],
    },
    {
        "name": "Order summary before CreateOrder",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Restaurant is open. Customer is Usman (existing)."},
            {"role": "user",   "content": "1 chicken biryani delivery chahiye, cash payment"},
        ],
        "expect_tool": None,   # should ask for confirmation first, not jump to CreateOrder
        "expect_keywords": ["biryani", "350", "delivery"],
    },
    {
        "name": "EasyPaisa payment — ask for screenshot",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Order total is Rs.730. EasyPaisa: 0300-1234567."},
            {"role": "user",   "content": "online pay karna chahta hun"},
        ],
        "expect_tool": None,
        "expect_keywords": ["easypaisa", "0300", "screenshot"],
    },
    {
        "name": "Out of scope redirect",
        "messages": [
            {"role": "system", "content": "You are DesiFood WhatsApp ordering assistant."},
            {"role": "user",   "content": "Pakistan ka cricket match kya hua?"},
        ],
        "expect_tool": None,
        "expect_keywords": ["order", "menu", "food", "restaurant"],
    },
    {
        "name": "Order modification request",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Customer Usman has Order #101 in 'Preparing' status."},
            {"role": "user",   "content": "order mein extra raita add karo"},
        ],
        "expect_tool": "CreateOrderModificationRequest",
        "expect_keywords": [],
    },
    {
        "name": "Short break response",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Restaurant IsOffline=1 (short break)."},
            {"role": "user",   "content": "order karna hai"},
        ],
        "expect_tool": None,
        "expect_keywords": ["break", "thori", "wait", "shortly"],
    },
    {
        "name": "Repeat last order",
        "messages": [
            {"role": "system", "content": "You are DesiFood assistant. Restaurant is open."},
            {"role": "user",   "content": "same as last time order karo"},
        ],
        "expect_tool": "GetUserOrder",
        "expect_keywords": [],
    },
]


def run_inference(model, tokenizer, messages: list, max_new_tokens=256) -> str:
    from unsloth import FastLanguageModel
    import torch

    FastLanguageModel.for_inference(model)

    # Convert system/user messages through our converter
    converted = [convert_message(m) for m in messages]

    inputs = tokenizer.apply_chat_template(
        converted, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True, top_p=0.9,
            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def validate(model, tokenizer) -> dict:
    print("\n" + "═"*60)
    print("  🔍 VALIDATION")
    print("═"*60)

    results = []
    for i, vc in enumerate(VALIDATION_CASES, 1):
        print(f"\n  [{i}/{len(VALIDATION_CASES)}] {vc['name']}")
        resp = run_inference(model, tokenizer, vc["messages"])
        resp_lower = resp.lower()
        print(f"  Response: {resp[:180]}{'...' if len(resp)>180 else ''}")

        scores = {}

        # Tool call check
        has_tool = "<tool_call>" in resp or '"name"' in resp
        if vc["expect_tool"]:
            hit = vc["expect_tool"].lower() in resp.lower()
            scores["tool"] = 1.0 if hit else 0.0
            print(f"  Tool [{vc['expect_tool']}]: {'✅' if hit else '❌'}")
        else:
            # Should NOT call a tool — if it does, penalize
            if has_tool and vc.get("expect_keywords"):
                scores["tool"] = 0.5
                print("  No tool expected but tool called ⚠️")
            else:
                scores["tool"] = 1.0
                print("  No tool (correct) ✅")

        # Keyword check
        if vc["expect_keywords"]:
            hits = [kw for kw in vc["expect_keywords"] if kw in resp_lower]
            scores["keywords"] = len(hits) / len(vc["expect_keywords"])
            print(f"  Keywords [{len(hits)}/{len(vc['expect_keywords'])}]: {hits}")
        else:
            scores["keywords"] = 1.0

        # Reasoning check
        scores["reasoning"] = 1.0 if "<think>" in resp else 0.0
        print(f"  Reasoning trace: {'✅' if scores['reasoning'] else '—'}")

        # Length sanity
        wc = len(resp.split())
        scores["length"] = 0.0 if wc < 5 else (0.7 if wc > 400 else 1.0)

        overall = sum(scores.values()) / len(scores)
        scores["overall"] = overall
        print(f"  Score: {overall:.0%}")
        results.append({"name": vc["name"], "scores": scores, "response": resp})

    avg = sum(r["scores"]["overall"] for r in results) / len(results)
    print("\n" + "═"*60)
    print(f"  📈 AVERAGE: {avg:.0%}  ", end="")
    if   avg >= 0.85: print("🌟 Excellent")
    elif avg >= 0.70: print("✅ Good")
    elif avg >= 0.50: print("⚠️  Fair — more data recommended")
    else:             print("❌ Poor — check dataset & config")
    print("═"*60)

    return {"average": avg, "results": results}


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export(model, tokenizer, output_dir, gguf=False):
    print(f"\n💾 Saving to: {output_dir}")
    lora = f"{output_dir}/lora_adapters"
    model.save_pretrained(lora); tokenizer.save_pretrained(lora)
    print(f"  ✅ LoRA adapters: {lora}")

    merged = f"{output_dir}/merged_16bit"
    model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
    print(f"  ✅ Merged model : {merged}")

    if gguf:
        gguf_path = f"{output_dir}/gguf_q4km"
        print("  🔄 Exporting GGUF (q4_k_m)...")
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"  ✅ GGUF: {gguf_path}")
        print(f"\n  Ollama usage:")
        print(f"    ollama create desifood -f {gguf_path}/Modelfile")
        print(f"    ollama run desifood")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="llama-3.2-3b", choices=list(MODELS.keys()))
    p.add_argument("--model-id",    default=None, help="Custom HF model ID")
    p.add_argument("--dataset",     default="restaurant_finetune_dataset_v2.jsonl")
    p.add_argument("--val-split",   type=float, default=0.15)
    p.add_argument("--epochs",      type=int,   default=TRAIN_CFG["num_epochs"])
    p.add_argument("--lr",          type=float, default=TRAIN_CFG["learning_rate"])
    p.add_argument("--batch-size",  type=int,   default=TRAIN_CFG["per_device_train_batch_size"])
    p.add_argument("--lora-r",      type=int,   default=TRAIN_CFG["lora_r"])
    p.add_argument("--max-seq-len", type=int,   default=TRAIN_CFG["max_seq_length"])
    p.add_argument("--bf16",        action="store_true")
    p.add_argument("--test-run",    action="store_true")
    p.add_argument("--skip-train",  action="store_true")
    p.add_argument("--validate",    action="store_true", help="Run validation (on by default unless --skip-train)")
    p.add_argument("--export-gguf", action="store_true")
    p.add_argument("--output-dir",  default=None)
    args = p.parse_args()

    model_id, max_seq = (args.model_id, args.max_seq_len) if args.model_id else MODELS[args.model]

    cfg = {**TRAIN_CFG}
    cfg.update({"num_epochs": args.epochs, "learning_rate": args.lr,
                "per_device_train_batch_size": args.batch_size,
                "lora_r": args.lora_r, "lora_alpha": args.lora_r * 2,
                "max_seq_length": args.max_seq_len})
    if args.bf16:
        cfg["bf16"], cfg["fp16"] = True, False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or f"./desifood_model_{args.model}_{ts}"
    os.makedirs(out, exist_ok=True)

    print("═"*60)
    print("  🍛 DesiFood — Unsloth Fine-Tuning v2")
    print("═"*60)
    print(f"  Model    : {model_id}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epochs   : {cfg['num_epochs']} | LR: {cfg['learning_rate']} | LoRA r: {cfg['lora_r']}")
    print(f"  Test run : {args.test_run}")
    print("═"*60)

    model, tokenizer = load_model(model_id, max_seq, cfg)

    if not args.skip_train:
        if not Path(args.dataset).exists():
            print(f"\n❌ Dataset not found: {args.dataset}")
            print("   Run generate_dataset_v2.py or generate_dataset_ai_v2.py first.")
            return

        train_ds, val_ds = load_dataset(args.dataset, args.val_split, tokenizer, max_seq)

        # Show sample formatted text
        print(f"\n--- Sample formatted training text (first 300 chars) ---")
        print(train_ds[0]["text"][:300])
        print("---\n")

        trainer, result = train(model, tokenizer, train_ds, val_ds, out, cfg, args.test_run)

        with open(f"{out}/training_config.json", "w") as f:
            json.dump({"model_id": model_id, "dataset": args.dataset,
                       "cfg": cfg, "final_loss": result.training_loss,
                       "format": "native_tool_calls_v2", "timestamp": ts}, f, indent=2)

    run_val = args.validate or not args.skip_train
    if run_val:
        val_results = validate(model, tokenizer)
        with open(f"{out}/validation_results.json", "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        print(f"  Validation saved: {out}/validation_results.json")

    if not args.skip_train:
        export(model, tokenizer, out, args.export_gguf)

    print(f"\n🎉 Complete! Output: {out}\n")


if __name__ == "__main__":
    main()