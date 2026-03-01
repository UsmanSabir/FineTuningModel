# 🍝 Bella Italia — Restaurant AI Agent Fine-Tuning Dataset

## Overview
A fine-tuning dataset for a restaurant AI agent with **tool calling**, **chain-of-thought thinking**, **customer support**, and **order management** capabilities.

---

## 📁 Files
| File | Description |
|------|-------------|
| `restaurant_finetune_dataset.jsonl` | Main training dataset (25 examples) |
| `generate_dataset.py` | Script to regenerate/extend the dataset |
| `README.md` | This documentation |

---

## 🧠 Dataset Design Principles

### 1. Thinking Traces (`<think>` tags)
Every non-trivial response includes a reasoning step inside `<think>...</think>` before acting. This teaches the model to:
- Identify what the customer actually needs
- Decide which tools to call (and in what order)
- Catch edge cases before they become problems

### 2. Tool Calling Format
```json
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>
```
Tool results are returned as `{"role": "tool", "content": "..."}` messages.

### 3. Multi-turn Conversations
Many examples span multiple turns to teach the model:
- Progressive information gathering
- Graceful clarification without frustration
- Maintaining context across the conversation

---

## 🛠️ Tools Included

| Tool | Purpose |
|------|---------|
| `get_menu` | Fetch menu by category |
| `create_order` | Place new orders |
| `get_order_status` | Track existing orders |
| `modify_order` | Add/remove/update items |
| `cancel_order` | Cancel pending orders |
| `check_table_availability` | Check reservation slots |
| `make_reservation` | Book a table |
| `process_refund` | Handle refund requests |
| `get_customer_history` | Retrieve past orders |
| `apply_discount` | Apply promo codes |

---

## 📊 Coverage by Category

| Category | Examples | Key Skills Taught |
|----------|----------|-------------------|
| Menu Browsing | 3 | Tool calling, allergy handling, recommendations |
| Order Placement | 4 | Info gathering, validation, confirmation |
| Order Tracking | 2 | Status interpretation, empathetic delays |
| Order Modification | 2 | Pre/post-kitchen logic, edge cases |
| Cancellations & Refunds | 2 | Full/partial refunds, policy handling |
| Reservations | 2 | Availability check + booking flow |
| Promos & Discounts | 2 | Success and error code handling |
| Customer History | 1 | Personalization, loyalty |
| Complex Multi-step | 2 | Chained tool calls, conditional logic |
| Edge Cases | 4 | Out-of-scope queries, angry customers, complaints |

---

## 🔧 How to Extend the Dataset

Run the generator and add more entries to the `dataset` list in `generate_dataset.py`:

```python
dataset.append({
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Your customer message here"},
        {"role": "assistant", "content": "<think>\nYour reasoning here\n</think>\n\nResponse with optional tool calls"},
        {"role": "tool", "content": json.dumps({"status": "success", ...})},
        {"role": "assistant", "content": "Final response to customer"}
    ]
})
```

### Recommended additional scenarios to add:
- Split bills / group orders
- Dietary combinations (vegan + nut allergy)
- No-show reservation policy
- Wait time queries during peak hours
- Upselling (recommending wine pairings)
- Feedback collection post-meal
- Gift cards / corporate orders
- Seasonal specials
- Large party/event bookings
- Language variations (non-native English speakers)

---

## 🎯 Fine-Tuning Recommendations

### Model Selection
- **Base:** `mistral-7b`, `llama-3-8b`, or `qwen2.5-7b` for lightweight agents
- **Premium:** `llama-3-70b` or `mixtral-8x7b` for complex reasoning

### Training Settings (suggested)
```yaml
learning_rate: 2e-5
epochs: 3-5
batch_size: 4
max_seq_length: 4096
lora_r: 16
lora_alpha: 32
```

### Format Compatibility
- **OpenAI-compatible:** The JSONL format matches the OpenAI fine-tuning spec
- Works with: **Axolotl**, **LLaMA-Factory**, **Unsloth**, **OpenAI fine-tuning API**

### Validation Split
Use 80/20 train/validation split. With 25 examples, consider expanding to 100+ before training.

---

## ✅ Quality Checklist

- [x] Every tool call has a corresponding tool result
- [x] Thinking traces are logical and step-by-step  
- [x] Error cases are handled gracefully
- [x] Allergy/safety warnings are included where appropriate
- [x] Emotional tone matches customer sentiment
- [x] Out-of-scope queries are declined politely
- [x] Multi-step flows have proper state continuity
