"""
═══════════════════════════════════════════════════════════════════
  Restaurant AI Agent — AI-Powered Dataset Generator v2
  Produces native OpenAI tool_calls format (matches LM Studio logs)
═══════════════════════════════════════════════════════════════════

Usage:
  python generate_dataset_ai_v2.py --provider ollama --model llama3.2
  python generate_dataset_ai_v2.py --provider openai --model gpt-4o --api-key sk-...
  python generate_dataset_ai_v2.py --dry-run
"""

import json, time, random, string, argparse, os
from typing import Optional

# ─────────────────────────────────────────────────────────────
# TOOLS (native format — matches LM Studio request body)
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {"type":"function","function":{"name":"GetCustomerDetails","description":"Get existing customer details (Id, Name, Number, Address, MapLink, Unsubscribed).","parameters":{"type":"object","properties":{},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"Restaurant_Status","description":"Get restaurant open/closed status, EasyPaisa number, min order amount, FAQ.","parameters":{"type":"object","properties":{},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"Read_Menu_Tool","description":"Fetch restaurant menu with food items, prices, and availability.","parameters":{"type":"object","properties":{},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"GetUserOrder","description":"Get customer orders with status and complete order data.","parameters":{"type":"object","properties":{"input":{"type":"string"}},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"GetLocation","description":"Get coordinates for a DHA Lahore address.","parameters":{"type":"object","properties":{"Phase":{"type":"number"},"Sector":{"type":"string"},"House":{"type":"string"},"SocietyName":{"type":"string"},"IsCommercial":{"type":"boolean"}},"required":["Phase","Sector","House","SocietyName","IsCommercial"],"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"GetDeliveryCharges","description":"Calculate delivery charges based on order amount and coordinates.","parameters":{"type":"object","properties":{"order_amount":{"type":"number"},"latitude":{"type":"string"},"longitude":{"type":"string"}},"required":["order_amount","latitude","longitude"],"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"CreateOrder","description":"Create a confirmed customer order after all details are verified.","parameters":{"type":"object","properties":{"delivery_charges":{"type":"number"},"total_amount":{"type":"string"},"payment_mode":{"type":"string"},"customer_name":{"type":"string"},"delivery_address":{"type":"string"},"latitude":{"type":"string"},"longitude":{"type":"string"},"quantity_ordered":{"type":"string"},"notes":{"type":"string"},"delivery_or_pickup":{"type":"string"},"payment_screenshot_id":{"type":"string"},"food_items":{"type":"string"},"order_amount":{"type":"number"}},"required":["delivery_charges","total_amount","payment_mode","customer_name","delivery_address","latitude","longitude","quantity_ordered","notes","delivery_or_pickup","payment_screenshot_id","food_items","order_amount"],"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"CreateOrderModificationRequest","description":"Submit an order modification request to the chef.","parameters":{"type":"object","properties":{"OrderId":{"type":"number"},"Request":{"type":"string"}},"required":["OrderId","Request"],"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"GetOrderModificationRequests","description":"Get customer's modification requests. Status 1=approved, 0=rejected.","parameters":{"type":"object","properties":{},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"SendOrderNowMessage","description":"Send the online order link to a customer who wants to order via website.","parameters":{"type":"object","properties":{"input":{"type":"string"}},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
    {"type":"function","function":{"name":"GetWhatsAppChatHistory","description":"Get prior WhatsApp chat history to resolve out-of-context messages.","parameters":{"type":"object","properties":{"input":{"type":"string"}},"additionalProperties":False,"$schema":"http://json-schema.org/draft-07/schema#"},"strict":False}},
]

SYSTEM_PROMPT = """# SYSTEM PROMPT — DESI FOOD WHATSAPP ORDER ASSISTANT

You are an AI ordering assistant for **DesiFood** restaurant operating via WhatsApp.

## STARTUP FLOW (every conversation)
1. Call `GetCustomerDetails` immediately — no greeting before this
2. Call `Restaurant_Status` — before ANY order handling
3. Call `Read_Menu_Tool` — at the start of each order session

## CUSTOMER HANDLING
- Existing customer → greet by saved name, then call `GetUserOrder` for default order context
- New customer → greet politely, collect name and address before ordering

## RESTAURANT STATUS RULES
- Open & Active → proceed with ordering
- Closed → politely inform, give opening time, offer website
- IsOffline=1 / On Break → "We are on a short break, please try again shortly"

## ORDER FLOW
1. Read menu → confirm items and quantities
2. Confirm delivery or pickup
3. Delivery → call `GetLocation` (DHA only) then `GetDeliveryCharges`
4. Show full order summary (items, total, delivery charges)
5. Ask payment: cash or online
6. Online → share EasyPaisa number, wait for screenshot ID
7. Call `CreateOrder` only after ALL details confirmed

## STYLE
- Short, friendly, WhatsApp-style messages
- Use customer's first name
- Urdu/English mix is acceptable"""

# ─────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────

SCENARIOS = [
    # Startup + status
    {"category": "startup",   "description": "Customer greets with 'AOA' — run full startup flow, restaurant is open, greet returning customer by name and ask what they want to order", "turns": 4},
    {"category": "startup",   "description": "Customer says 'menu dikhao' — startup flow, restaurant open, show full menu in WhatsApp-friendly format", "turns": 4},
    {"category": "closed",    "description": "Customer wants to order but restaurant is closed — inform politely with opening hours, offer website link", "turns": 3},
    {"category": "break",     "description": "Customer wants to order but IsOffline=1 (short break) — inform politely and ask to try again shortly", "turns": 3},

    # Order flows
    {"category": "order",     "description": "Customer orders 1 chicken biryani + 2 naan for delivery, cash payment, existing customer with saved DHA address", "turns": 8},
    {"category": "order",     "description": "Customer orders beef karahi for delivery, wants to pay via EasyPaisa online, sends screenshot", "turns": 9},
    {"category": "order",     "description": "Customer orders daal makhani and naan for pickup — no location/delivery charges needed", "turns": 6},
    {"category": "order",     "description": "New customer places their first order — agent collects name and address, explains payment options", "turns": 8},
    {"category": "order",     "description": "Customer wants to order but the item they ask for (mutton biryani) is marked unavailable — agent suggests alternatives", "turns": 5},
    {"category": "order",     "description": "Customer asks about minimum order amount before ordering — agent explains from Restaurant_Status FAQ", "turns": 3},
    {"category": "order",     "description": "Customer orders multiple items, wants to confirm delivery time before confirming payment", "turns": 7},
    {"category": "order",     "description": "Returning customer says 'same as last time' — agent uses GetUserOrder to find and repeat last order", "turns": 6},

    # Order tracking & modification
    {"category": "tracking",  "description": "Customer asks 'mera order kahan hai' — agent checks GetUserOrder and reports status", "turns": 4},
    {"category": "tracking",  "description": "Customer's order shows 'Delivered' but they say they haven't received it — agent empathizes and escalates", "turns": 4},
    {"category": "modify",    "description": "Customer wants to add extra raita to an order that's already being prepared — submit modification request", "turns": 5},
    {"category": "modify",    "description": "Customer asks about status of their modification request — request was approved", "turns": 4},
    {"category": "modify",    "description": "Customer asks about modification request — it was rejected by chef with reason", "turns": 4},

    # Context & chat history
    {"category": "context",   "description": "Customer sends an out-of-context message 'haan same wala' — agent calls GetWhatsAppChatHistory to understand and responds correctly", "turns": 5},
    {"category": "context",   "description": "Customer references a previous conversation — agent uses chat history to understand and assist", "turns": 5},

    # Support & edge cases
    {"category": "support",   "description": "Customer having difficulty ordering on WhatsApp — agent sends order link via SendOrderNowMessage", "turns": 4},
    {"category": "support",   "description": "Customer complains their food was cold — agent apologizes and offers resolution", "turns": 4},
    {"category": "support",   "description": "Customer asks what areas you deliver to — agent answers from Restaurant_Status info", "turns": 3},
    {"category": "support",   "description": "Customer asks if they can order for tomorrow (advance order) — agent explains policy", "turns": 3},
    {"category": "edge",      "description": "Customer asks completely unrelated question (cricket score) — agent politely redirects", "turns": 2},
    {"category": "edge",      "description": "Customer is rude and frustrated about a late order — agent stays calm, professional, and helpful", "turns": 5},
    {"category": "edge",      "description": "Customer messages in mix of Urdu and English — agent responds naturally in same style", "turns": 6},
    {"category": "edge",      "description": "Customer provides an address that is NOT in DHA — GetLocation fails, agent handles gracefully", "turns": 5},
]

# ─────────────────────────────────────────────────────────────
# GENERATOR PROMPT
# ─────────────────────────────────────────────────────────────

GENERATOR_PROMPT = """You are building fine-tuning training data for a WhatsApp restaurant ordering AI.

Generate a realistic multi-turn conversation for this scenario:
SCENARIO: {description}
CATEGORY: {category}
EXPECTED TURNS: ~{turns}

## CRITICAL FORMAT RULES

The output must be a JSON object with exactly two top-level keys: "tools" and "messages".

### Tool call format (assistant calling a tool):
```json
{{"role": "assistant", "content": "", "reasoning": "step by step thinking here", "tool_calls": [{{"type": "function", "id": "123456789", "function": {{"name": "ToolName", "arguments": "{{}}"}}}}]}}
```
- content MUST be empty string "" when tool_calls is present
- reasoning MUST explain WHY this tool is being called
- id should be a random 9-digit number string

### Tool result format:
```json
{{"role": "tool", "content": "{{...json result...}}", "name": "ToolName", "tool_call_id": "123456789"}}
```
- tool_call_id MUST match the id from the corresponding tool_calls entry
- name MUST match the tool function name

### Assistant reply format (when responding to user, no tool):
```json
{{"role": "assistant", "content": "message to user", "reasoning": "optional reasoning", "tool_calls": []}}
```

### STARTUP FLOW (mandatory for every conversation):
Turn 1: assistant calls GetCustomerDetails (no greeting before this)
Turn 2: assistant calls Restaurant_Status  
Turn 3: assistant calls Read_Menu_Tool (if placing an order)
Only THEN greet the customer and proceed.

### Tool ID rule: Each tool call gets a unique random 9-digit ID. The tool result must reference the SAME ID.

## OUTPUT

Return ONLY the raw JSON. No markdown fences. No explanation.

Structure:
{{
  "tools": {tools},
  "messages": [
    {{"role": "system", "content": {system_prompt}}},
    ... conversation turns ...
  ]
}}

System prompt to use verbatim (copy exactly):
---
{system_prompt}
---

Generate the conversation now:"""

# ─────────────────────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────────────────────

def call_openai_compatible(messages, model, base_url, api_key, temperature=0.7, max_tokens=6000):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")
    client = OpenAI(api_key=api_key, base_url=base_url)
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content


def call_ollama(messages, model, base_url="http://localhost:11434", temperature=0.7, max_tokens=6000):
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")
    r = requests.post(f"{base_url}/api/chat", json={
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_example(ex: dict) -> tuple[bool, list]:
    """Validate structure matches native tool_calls format. Returns (valid, errors)."""
    errors = []

    if "tools" not in ex:
        errors.append("Missing top-level 'tools' key")
    if "messages" not in ex:
        errors.append("Missing top-level 'messages' key")
        return False, errors

    msgs = ex["messages"]
    if len(msgs) < 4:
        errors.append(f"Too few messages: {len(msgs)}")

    if msgs[0].get("role") != "system":
        errors.append("First message must be role=system")

    # Check tool_calls / tool result pairing
    pending_calls = {}
    for i, m in enumerate(msgs):
        role = m.get("role")

        if role == "assistant" and m.get("tool_calls"):
            # Validate tool call structure
            for tc in m["tool_calls"]:
                tc_id = tc.get("id")
                fn = tc.get("function", {})
                if not tc_id:
                    errors.append(f"Message {i}: tool_call missing 'id'")
                if not fn.get("name"):
                    errors.append(f"Message {i}: tool_call missing function.name")
                if tc_id:
                    pending_calls[tc_id] = fn.get("name", "?")
            if m.get("content") not in ("", None):
                errors.append(f"Message {i}: assistant tool_call message should have empty content")

        elif role == "tool":
            tc_id = m.get("tool_call_id")
            name  = m.get("name")
            if not tc_id:
                errors.append(f"Message {i}: tool result missing 'tool_call_id'")
            if not name:
                errors.append(f"Message {i}: tool result missing 'name'")
            if tc_id and tc_id not in pending_calls:
                errors.append(f"Message {i}: tool_call_id '{tc_id}' has no matching tool call")
            if tc_id in pending_calls:
                del pending_calls[tc_id]

    if pending_calls:
        errors.append(f"Unmatched tool calls (no result): {list(pending_calls.values())}")

    return len(errors) == 0, errors

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def generate_example(scenario, provider, model, base_url, api_key, max_retries=3):
    prompt = GENERATOR_PROMPT.format(
        description=scenario["description"],
        category=scenario["category"],
        turns=scenario["turns"],
        tools=json.dumps(TOOLS, ensure_ascii=False),
        system_prompt=json.dumps(SYSTEM_PROMPT)
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries}...", end=" ", flush=True)

            raw = call_ollama([{"role": "user", "content": prompt}], model, base_url) \
                  if provider == "ollama" \
                  else call_openai_compatible([{"role": "user", "content": prompt}], model, base_url, api_key)

            print(f"Parsing response...{raw} (response_end)", end=" ", flush=True)
            # Strip markdown fences
            raw = raw.strip()
            for fence in ["```json", "```"]:
                if raw.startswith(fence):
                    raw = raw[len(fence):]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            parsed = json.loads(raw)

            valid, errors = validate_example(parsed)
            if not valid:
                print(f"❌ Validation: {errors[:2]}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            print("✅")
            return parsed

        except json.JSONDecodeError as e:
            print(f"❌ JSON: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return None

# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────

def report_stats(dataset):
    total_turns  = sum(len(ex["messages"]) for ex in dataset)
    tool_calls   = sum(1 for ex in dataset for m in ex["messages"] if m.get("tool_calls"))
    tool_results = sum(1 for ex in dataset for m in ex["messages"] if m["role"] == "tool")
    with_reason  = sum(1 for ex in dataset for m in ex["messages"] if m.get("reasoning"))
    categories   = {}
    for ex in dataset:
        c = ex.get("_meta", {}).get("category", "unknown")
        categories[c] = categories.get(c, 0) + 1

    print("\n" + "═"*52)
    print("  📊 DATASET STATISTICS")
    print("═"*52)
    print(f"  Examples              : {len(dataset)}")
    print(f"  Total message turns   : {total_turns}")
    print(f"  tool_calls[] turns    : {tool_calls}")
    print(f"  tool result turns     : {tool_results}")
    print(f"  reasoning field turns : {with_reason}")
    print(f"\n  Category breakdown:")
    for cat, n in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {'█'*n} ({n})")
    print("═"*52)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",  choices=["openai","ollama"], default="ollama")
    parser.add_argument("--model",     default="llama3.2")
    parser.add_argument("--api-key",   default=os.environ.get("OPENAI_API_KEY","ollama"))
    parser.add_argument("--base-url",  default=None)
    parser.add_argument("--output",    default="restaurant_finetune_dataset_ai_v2.jsonl")
    parser.add_argument("--count",     type=int, default=len(SCENARIOS))
    parser.add_argument("--shuffle",   action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--delay",     type=float, default=1.0)
    args = parser.parse_args()

    if args.base_url is None:
        args.base_url = "http://localhost:11434" if args.provider == "ollama" else "https://api.openai.com/v1"

    print("═"*60)
    print("  🍛 DesiFood — AI Dataset Generator v2 (Native tool_calls)")
    print("═"*60)
    print(f"  Provider  : {args.provider} | Model: {args.model}")
    print(f"  Scenarios : {args.count} of {len(SCENARIOS)}")
    print(f"  Output    : {args.output}")
    print("═"*60)

    scenarios = SCENARIOS[:args.count]
    if args.shuffle:
        random.shuffle(scenarios)

    if args.dry_run:
        print("\n📋 Scenarios:\n")
        for i, s in enumerate(scenarios, 1):
            print(f"  {i:>2}. [{s['category']:<12}] {s['description'][:80]}")
        print(f"\nTotal: {len(scenarios)}. Remove --dry-run to generate.")
        return

    dataset, failed = [], []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i:>2}/{len(scenarios)}] [{scenario['category'].upper()}] {scenario['description'][:70]}")

        ex = generate_example(scenario, args.provider, args.model, args.base_url, args.api_key)
        if ex:
            ex["_meta"] = {"category": scenario["category"], "scenario": scenario["description"]}
            dataset.append(ex)
            with open(args.output, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            failed.append(scenario)

        if i < len(scenarios):
            time.sleep(args.delay)

    report_stats(dataset)
    if failed:
        print(f"\n⚠️  {len(failed)} failed:")
        for s in failed:
            print(f"   - {s['description'][:60]}")
    print(f"\n✅ Saved: {args.output} ({len(dataset)} examples)\n")


if __name__ == "__main__":
    main()
