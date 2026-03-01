"""
═══════════════════════════════════════════════════════════════════
  Restaurant AI Agent — Dataset Generator (AI-Powered)
  Supports: OpenAI, Groq, Together AI, LM Studio, Ollama
═══════════════════════════════════════════════════════════════════

Usage:
  # OpenAI / Groq / Together / LM Studio (OpenAI-compatible)
  python generate_dataset_ai.py --provider openai --model gpt-4o --api-key sk-...

  # Ollama (local)
  python generate_dataset_ai.py --provider ollama --model llama3.2

  # Custom OpenAI-compatible endpoint
  python generate_dataset_ai.py --provider openai --base-url http://localhost:1234/v1 --model local-model

  # Generate more examples
  python generate_dataset_ai.py --provider ollama --model llama3.2 --count 50

  # Dry run — prints scenarios without calling API
  python generate_dataset_ai.py --dry-run
"""

import json
import time
import random
import argparse
import os
from typing import Optional

# ─────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (same as in main dataset)
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {"name": "get_menu", "description": "Fetch the current menu. Optionally filter by category.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "enum": ["appetizers", "mains", "desserts", "drinks", "all"]}}, "required": ["category"]}},
    {"name": "create_order", "description": "Place a new food order for a customer.", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "properties": {"item_id": {"type": "string"}, "name": {"type": "string"}, "quantity": {"type": "integer"}, "special_instructions": {"type": "string"}}, "required": ["item_id", "name", "quantity"]}}, "order_type": {"type": "string", "enum": ["dine_in", "takeout", "delivery"]}, "delivery_address": {"type": "string"}}, "required": ["customer_name", "items", "order_type"]}},
    {"name": "get_order_status", "description": "Check the status of an existing order by order ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "modify_order", "description": "Modify an existing order before it is prepared.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "action": {"type": "string", "enum": ["add_item", "remove_item", "update_quantity"]}, "item": {"type": "object", "properties": {"item_id": {"type": "string"}, "name": {"type": "string"}, "quantity": {"type": "integer"}}}}, "required": ["order_id", "action", "item"]}},
    {"name": "cancel_order", "description": "Cancel an order if it hasn't been prepared yet.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}},
    {"name": "check_table_availability", "description": "Check available tables for a given date, time, and party size.", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "party_size": {"type": "integer"}}, "required": ["date", "time", "party_size"]}},
    {"name": "make_reservation", "description": "Reserve a table for a customer.", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string"}, "phone": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "party_size": {"type": "integer"}, "special_requests": {"type": "string"}}, "required": ["customer_name", "phone", "date", "time", "party_size"]}},
    {"name": "process_refund", "description": "Initiate a refund for a completed order.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}, "refund_amount": {"type": "number"}}, "required": ["order_id", "reason", "refund_amount"]}},
    {"name": "get_customer_history", "description": "Retrieve past orders and preferences for a customer.", "parameters": {"type": "object", "properties": {"customer_name": {"type": "string"}, "phone": {"type": "string"}}, "required": ["customer_name"]}},
    {"name": "apply_discount", "description": "Apply a promo code or loyalty discount to an order.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "promo_code": {"type": "string"}}, "required": ["order_id", "promo_code"]}},
]

SYSTEM_PROMPT = f"""You are Bella, a friendly and efficient AI agent for Bella Italia Restaurant. You help customers with:
- Browsing the menu and making recommendations
- Placing, modifying, and tracking orders
- Making and managing reservations
- Handling complaints, refunds, and special requests

You have access to the following tools:
{json.dumps(TOOLS, indent=2)}

When you need to call a tool, respond with a JSON block in this format:
<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>

After receiving a tool result, use it to craft a helpful, warm response.

THINKING: Before responding to complex requests, think step by step inside <think>...</think> tags. Identify:
1. What the customer actually needs
2. Which tools are required and in what order
3. Any edge cases or clarifications needed

Always be warm, concise, and solution-focused."""

# ─────────────────────────────────────────────────────────────
# SCENARIOS — Feed these to the AI to generate conversations
# ─────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── Menu & Recommendations ──
    {"category": "menu", "description": "Customer asks for the best dish for a first date", "turns": 2},
    {"category": "menu", "description": "Customer wants to know calorie info for the pasta dishes", "turns": 2},
    {"category": "menu", "description": "Customer is vegan and asks what they can eat", "turns": 2},
    {"category": "menu", "description": "Customer has a nut allergy and wants safe options", "turns": 2},
    {"category": "menu", "description": "Customer asks what's new or seasonal on the menu", "turns": 2},
    {"category": "menu", "description": "Customer wants a wine pairing recommendation with their salmon", "turns": 3},
    {"category": "menu", "description": "Customer asks what kids meal options are available", "turns": 2},
    {"category": "menu", "description": "Customer asks about portion sizes — are they big enough for sharing?", "turns": 2},

    # ── Order Placement ──
    {"category": "order", "description": "Customer places a large group delivery order for 8 people with multiple dietary needs", "turns": 4},
    {"category": "order", "description": "Customer wants to order but is unsure between two dishes — asks for help deciding", "turns": 3},
    {"category": "order", "description": "Customer asks to place a scheduled order for 6pm tonight (not immediate)", "turns": 3},
    {"category": "order", "description": "Customer wants to split an order — half takeout for now, wants dessert later dine-in", "turns": 4},
    {"category": "order", "description": "Customer orders for the first time and forgets to give their address for delivery", "turns": 3},
    {"category": "order", "description": "Customer wants extra sauce, no onions, well-done steak — custom order with special instructions", "turns": 3},
    {"category": "order", "description": "Customer tries to order an item that's not on the menu", "turns": 2},
    {"category": "order", "description": "Customer places an order then immediately wants to change the delivery address", "turns": 3},

    # ── Order Management ──
    {"category": "order_management", "description": "Customer's order status shows 'delayed' and they're getting impatient", "turns": 3},
    {"category": "order_management", "description": "Customer received wrong items in their order and wants it fixed", "turns": 4},
    {"category": "order_management", "description": "Customer wants to upgrade their takeout order to delivery after placing it", "turns": 3},
    {"category": "order_management", "description": "Customer asks to change quantity of an item already in a confirmed order", "turns": 3},
    {"category": "order_management", "description": "Customer wants to cancel but order is already being prepared", "turns": 3},
    {"category": "order_management", "description": "Customer's food arrived cold — they want either replacement or refund", "turns": 4},
    {"category": "order_management", "description": "Customer received food with an ingredient they're allergic to despite noting it", "turns": 4},

    # ── Reservations ──
    {"category": "reservation", "description": "Customer wants to book for a surprise birthday party of 12 with decoration setup", "turns": 4},
    {"category": "reservation", "description": "Customer calls to modify an existing reservation — wants to change time and add 2 more guests", "turns": 3},
    {"category": "reservation", "description": "Customer wants to cancel their reservation due to illness", "turns": 2},
    {"category": "reservation", "description": "Customer asks if they can bring their own birthday cake", "turns": 2},
    {"category": "reservation", "description": "No tables available at 7pm — agent offers alternatives", "turns": 3},
    {"category": "reservation", "description": "Customer wants to book a private dining room for a business lunch", "turns": 3},

    # ── Customer Support ──
    {"category": "support", "description": "Customer complains about rude service from a waiter and wants to escalate", "turns": 3},
    {"category": "support", "description": "Customer lost an item (phone) in the restaurant and wants help finding it", "turns": 2},
    {"category": "support", "description": "Customer asks about parking availability near the restaurant", "turns": 2},
    {"category": "support", "description": "Customer is unhappy with their food and the restaurant's response so far — escalation", "turns": 4},
    {"category": "support", "description": "Customer asks if the restaurant does catering for events", "turns": 2},
    {"category": "support", "description": "Customer asks about gift card purchasing options", "turns": 2},
    {"category": "support", "description": "Customer wants to leave a compliment for a specific staff member", "turns": 2},

    # ── Promos & Loyalty ──
    {"category": "promo", "description": "Customer asks how to earn and redeem loyalty points", "turns": 2},
    {"category": "promo", "description": "Customer wants to apply two promo codes to one order", "turns": 3},
    {"category": "promo", "description": "Customer's promo code is expired — agent offers alternative", "turns": 3},
    {"category": "promo", "description": "Returning customer asks for a loyalty discount as a long-time regular", "turns": 3},

    # ── Edge Cases ──
    {"category": "edge", "description": "Customer asks completely unrelated question (weather, sports score)", "turns": 1},
    {"category": "edge", "description": "Customer is rude and aggressive — agent maintains professionalism", "turns": 3},
    {"category": "edge", "description": "Customer asks about competitors' restaurants", "turns": 1},
    {"category": "edge", "description": "Non-native English speaker with broken grammar asks for help", "turns": 3},
    {"category": "edge", "description": "Customer keeps changing their mind mid-order multiple times", "turns": 5},
    {"category": "edge", "description": "Customer claims they were double-charged for an order", "turns": 4},
    {"category": "edge", "description": "Customer asks to speak to a human manager", "turns": 2},
]

# ─────────────────────────────────────────────────────────────
# GENERATOR PROMPT — what we tell the AI to produce
# ─────────────────────────────────────────────────────────────
GENERATOR_PROMPT_TEMPLATE = """You are creating fine-tuning training data for a restaurant AI agent called Bella.

Generate a realistic, multi-turn conversation for this scenario:
SCENARIO: {description}
CATEGORY: {category}
EXPECTED TURNS: approximately {turns} turns

The conversation must follow this EXACT JSON structure:
{{
  "messages": [
    {{"role": "system", "content": "<SYSTEM_PROMPT>"}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    // more turns as needed
    // when agent calls a tool, use:
    {{"role": "assistant", "content": "<think>\\nreasoning here\\n</think>\\n\\ntext + <tool_call>\\n{{...}}\\n</tool_call>"}},
    {{"role": "tool", "content": "{{...json result...}}"}},
    {{"role": "assistant", "content": "final response to customer"}}
  ]
}}

RULES:
1. Replace <SYSTEM_PROMPT> with the system prompt verbatim (copy it exactly from the context)
2. Assistant MUST use <think>...</think> for any non-trivial reasoning
3. Tool calls MUST be inside <tool_call>...</tool_call> blocks with valid JSON
4. Tool results must be realistic JSON responses
5. Order IDs: ORD-XXXX format, Reservation IDs: RES-XXXX, Refund IDs: REF-XXXX
6. Be warm, helpful, and professional. Acknowledge emotions when customers are upset.
7. The conversation must feel natural and human
8. Return ONLY the raw JSON object. No markdown code fences. No explanation text.

The system prompt to use verbatim:
---
{system_prompt}
---

Generate the conversation now:"""

# ─────────────────────────────────────────────────────────────
# CLIENT WRAPPERS
# ─────────────────────────────────────────────────────────────

def call_openai_compatible(
    messages: list,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Call any OpenAI-compatible API (OpenAI, Groq, Together, LM Studio, etc.)"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_ollama(
    messages: list,
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Call Ollama local API"""
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ─────────────────────────────────────────────────────────────
# GENERATION LOGIC
# ─────────────────────────────────────────────────────────────

def generate_example(
    scenario: dict,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int = 3,
) -> Optional[dict]:
    """Generate a single training example for a given scenario."""

    prompt = GENERATOR_PROMPT_TEMPLATE.format(
        description=scenario["description"],
        category=scenario["category"],
        turns=scenario["turns"],
        system_prompt=SYSTEM_PROMPT,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries}...", end=" ", flush=True)

            if provider == "ollama":
                raw = call_ollama(messages, model, base_url)
            else:
                raw = call_openai_compatible(messages, model, base_url, api_key)

            # Strip markdown code fences if the model added them
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[-1]          # remove opening fence
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0]           # remove closing fence
            raw = raw.strip()

            parsed = json.loads(raw)

            # Basic validation
            assert "messages" in parsed, "Missing 'messages' key"
            assert len(parsed["messages"]) >= 3, "Too few messages"
            assert parsed["messages"][0]["role"] == "system", "First message must be system"

            print("✅")
            return parsed

        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
        except AssertionError as e:
            print(f"❌ Validation error: {e}")
        except Exception as e:
            print(f"❌ API error: {e}")

        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    return None


# ─────────────────────────────────────────────────────────────
# STATS REPORTER
# ─────────────────────────────────────────────────────────────

def report_stats(dataset: list):
    total_turns = sum(len(ex["messages"]) for ex in dataset)
    tool_calls = sum(
        1 for ex in dataset
        for m in ex["messages"]
        if m["role"] == "assistant" and "<tool_call>" in m.get("content", "")
    )
    think_traces = sum(
        1 for ex in dataset
        for m in ex["messages"]
        if m["role"] == "assistant" and "<think>" in m.get("content", "")
    )
    categories = {}
    for ex in dataset:
        cat = ex.get("_meta", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "═" * 50)
    print("  📊 DATASET STATISTICS")
    print("═" * 50)
    print(f"  Total examples    : {len(dataset)}")
    print(f"  Total turns       : {total_turns}")
    print(f"  Tool call turns   : {tool_calls}")
    print(f"  Think trace turns : {think_traces}")
    print(f"\n  Category breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {cat:<20} {bar} ({count})")
    print("═" * 50)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate restaurant fine-tuning dataset using AI")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama",
                        help="API provider to use (default: ollama)")
    parser.add_argument("--model", default="llama3.2",
                        help="Model name (default: llama3.2 for Ollama, gpt-4o for OpenAI)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "ollama"),
                        help="API key (uses OPENAI_API_KEY env var if not set)")
    parser.add_argument("--base-url", default=None,
                        help="Custom base URL (e.g. http://localhost:11434 for Ollama, http://localhost:1234/v1 for LM Studio)")
    parser.add_argument("--output", default="restaurant_finetune_dataset_ai.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=len(SCENARIOS),
                        help=f"Number of examples to generate (max: {len(SCENARIOS)})")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle scenario order before generating")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scenarios without calling the API")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Generation temperature (default: 0.8)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds (default: 1.0)")
    args = parser.parse_args()

    # Set defaults for base URLs
    if args.base_url is None:
        args.base_url = "http://localhost:11434" if args.provider == "ollama" else "https://api.openai.com/v1"

    print("═" * 60)
    print("  🍝 Bella Italia — AI Dataset Generator")
    print("═" * 60)
    print(f"  Provider   : {args.provider}")
    print(f"  Model      : {args.model}")
    print(f"  Base URL   : {args.base_url}")
    print(f"  Output     : {args.output}")
    print(f"  Count      : {args.count} / {len(SCENARIOS)} scenarios")
    print(f"  Dry run    : {args.dry_run}")
    print("═" * 60 + "\n")

    scenarios = SCENARIOS[:args.count]
    if args.shuffle:
        random.shuffle(scenarios)

    if args.dry_run:
        print("📋 Scenarios to generate:\n")
        for i, s in enumerate(scenarios, 1):
            print(f"  {i:>2}. [{s['category']}] {s['description']}")
        print(f"\nTotal: {len(scenarios)} scenarios. Remove --dry-run to generate.")
        return

    dataset = []
    failed = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i:>2}/{len(scenarios)}] [{scenario['category'].upper()}] {scenario['description']}")

        example = generate_example(
            scenario=scenario,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_retries=3,
        )

        if example:
            example["_meta"] = {"category": scenario["category"], "scenario": scenario["description"]}
            dataset.append(example)

            # Save incrementally so progress isn't lost on failure
            with open(args.output, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
        else:
            failed.append(scenario)
            print(f"  ⚠️  Skipping after all retries failed.")

        if i < len(scenarios):
            time.sleep(args.delay)

    # Final report
    report_stats(dataset)

    if failed:
        print(f"\n  ⚠️  {len(failed)} scenarios failed:")
        for s in failed:
            print(f"     - [{s['category']}] {s['description']}")

    print(f"\n✅ Dataset saved to: {args.output}")
    print(f"   {len(dataset)} examples written.\n")


if __name__ == "__main__":
    main()
