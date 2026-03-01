"""
═══════════════════════════════════════════════════════════════════
  Restaurant AI Agent — Dataset Importer & Validator
  Imports JSON/JSONL, validates structure, appends to output JSONL
═══════════════════════════════════════════════════════════════════

Usage:
  # Import a JSON file (array of examples)
  python import_dataset.py --input my_data.json --output restaurant_finetune_dataset_v2.jsonl

  # Import a JSONL file
  python import_dataset.py --input more_data.jsonl --output restaurant_finetune_dataset_v2.jsonl

  # Dry run — validate only, don't write anything
  python import_dataset.py --input my_data.json --dry-run

  # Skip duplicate check (allow same conversations)
  python import_dataset.py --input my_data.json --output out.jsonl --no-dedup

  # Show detailed errors for each failed example
  python import_dataset.py --input my_data.json --verbose
"""

import json
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# VALIDATION LOGIC  (single source of truth — reused everywhere)
# Import this function in any other script instead of duplicating
# ─────────────────────────────────────────────────────────────

def validate_example(ex: dict) -> tuple[bool, list[str]]:
    """
    Validate a single training example against the native OpenAI
    tool_calls format used in this project.

    Returns:
        (is_valid: bool, errors: list[str])

    Reuse this function in:
        generate_dataset_v2.py
        generate_dataset_ai_v2.py
        train_unsloth_v2.py
        this script
    """
    errors = []

    # ── Top-level structure ──
    if not isinstance(ex, dict):
        return False, ["Example must be a JSON object, not " + type(ex).__name__]

    if "messages" not in ex:
        errors.append("Missing required key: 'messages'")
        return False, errors  # can't continue without messages

    if "tools" not in ex:
        errors.append("Missing recommended key: 'tools' (tools should be top-level, not in system prompt)")
        # non-fatal — continue validation

    messages = ex["messages"]

    # ── Messages list ──
    if not isinstance(messages, list):
        errors.append("'messages' must be an array")
        return False, errors

    if len(messages) < 3:
        errors.append(f"Too few messages: {len(messages)} (minimum 3: system + user + assistant)")

    # ── First message must be system ──
    if messages and messages[0].get("role") != "system":
        errors.append(f"First message must be role=system, got role={messages[0].get('role')!r}")

    if messages and not messages[0].get("content", "").strip():
        errors.append("System message content is empty")

    # ── Validate each message and track tool call IDs ──
    valid_roles = {"system", "user", "assistant", "tool"}
    pending_calls: dict[str, str] = {}   # id → tool_name (calls waiting for a result)
    seen_roles_after_system = []

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            errors.append(f"Message {i}: must be a JSON object")
            continue

        role = m.get("role")
        content = m.get("content")
        tool_calls = m.get("tool_calls", [])

        # Role check
        if role not in valid_roles:
            errors.append(f"Message {i}: unknown role {role!r} (allowed: {valid_roles})")
            continue

        if i > 0:
            seen_roles_after_system.append(role)

        # ── assistant message ──
        if role == "assistant":
            if tool_calls:
                # Tool-calling turn: content must be "" or None
                if content not in ("", None):
                    errors.append(
                        f"Message {i}: assistant tool_call message should have empty content, "
                        f"got {content!r:.40}"
                    )
                # Validate each tool call entry
                for j, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        errors.append(f"Message {i}, tool_call {j}: must be an object")
                        continue

                    tc_id   = tc.get("id")
                    tc_type = tc.get("type")
                    fn      = tc.get("function", {})
                    fn_name = fn.get("name")
                    fn_args = fn.get("arguments")

                    if tc_type != "function":
                        errors.append(f"Message {i}, tool_call {j}: type must be 'function', got {tc_type!r}")
                    if not tc_id:
                        errors.append(f"Message {i}, tool_call {j}: missing 'id'")
                    if not fn_name:
                        errors.append(f"Message {i}, tool_call {j}: missing function.name")
                    if fn_args is None:
                        errors.append(f"Message {i}, tool_call {j}: missing function.arguments")
                    elif isinstance(fn_args, str):
                        # arguments must be valid JSON string
                        try:
                            json.loads(fn_args)
                        except json.JSONDecodeError:
                            errors.append(
                                f"Message {i}, tool_call {j}: function.arguments is not valid JSON: {fn_args!r:.40}"
                            )
                    elif isinstance(fn_args, dict):
                        errors.append(
                            f"Message {i}, tool_call {j}: function.arguments should be a JSON string, not an object"
                        )

                    if tc_id and fn_name:
                        if tc_id in pending_calls:
                            errors.append(f"Message {i}: duplicate tool call id {tc_id!r}")
                        pending_calls[tc_id] = fn_name

            else:
                # Regular reply: must have non-empty content
                if not content:
                    errors.append(f"Message {i}: assistant reply has no tool_calls and empty content")

        # ── tool result message ──
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            name  = m.get("name")

            if not tc_id:
                errors.append(f"Message {i}: tool result missing 'tool_call_id'")
            if not name:
                errors.append(f"Message {i}: tool result missing 'name'")
            if content is None:
                errors.append(f"Message {i}: tool result missing 'content'")
            elif isinstance(content, str):
                # content should be valid JSON string
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    errors.append(
                        f"Message {i}: tool result content is not valid JSON: {content!r:.60}"
                    )

            # Match to pending call
            if tc_id:
                if tc_id not in pending_calls:
                    errors.append(
                        f"Message {i}: tool_call_id {tc_id!r} has no matching tool_call (orphaned result)"
                    )
                else:
                    # Name should match the tool call function name
                    expected_name = pending_calls.pop(tc_id)
                    if name and name != expected_name:
                        errors.append(
                            f"Message {i}: tool result name {name!r} doesn't match "
                            f"tool call name {expected_name!r}"
                        )

        # ── user message ──
        elif role == "user":
            if not content and not content == "":
                errors.append(f"Message {i}: user message missing content")

    # ── Unmatched tool calls (called but no result provided) ──
    if pending_calls:
        for tc_id, fn_name in pending_calls.items():
            errors.append(f"Unmatched tool call: {fn_name!r} (id={tc_id}) was called but has no tool result")

    # ── Conversation must end with assistant ──
    if messages and messages[-1].get("role") != "assistant":
        errors.append(
            f"Last message should be role=assistant, got {messages[-1].get('role')!r}"
        )

    # ── Tools array validation (if present) ──
    tools = ex.get("tools", [])
    if tools:
        if not isinstance(tools, list):
            errors.append("'tools' must be an array")
        else:
            for j, t in enumerate(tools):
                if not isinstance(t, dict):
                    errors.append(f"Tool {j}: must be an object")
                    continue
                if t.get("type") != "function":
                    errors.append(f"Tool {j}: type must be 'function'")
                fn = t.get("function", {})
                if not fn.get("name"):
                    errors.append(f"Tool {j}: missing function.name")
                if "parameters" not in fn:
                    errors.append(f"Tool {j} ({fn.get('name', '?')}): missing function.parameters")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────

def load_input_file(path: str) -> tuple[list[dict], list[str]]:
    """
    Load examples from a .json or .jsonl file.
    Returns (examples, load_errors).
    """
    p = Path(path)
    if not p.exists():
        return [], [f"File not found: {path}"]

    ext = p.suffix.lower()
    raw = p.read_text(encoding="utf-8").strip()

    examples = []
    load_errors = []

    if ext == ".jsonl" or (ext not in (".json",) and "\n{" in raw):
        # JSONL — one JSON object per line
        for i, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                load_errors.append(f"Line {i}: {e}")

    else:
        # JSON — either a single object or an array
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict):
                # Could be a single example or a wrapper {"data": [...]}
                if "messages" in data:
                    examples = [data]
                elif any(k in data for k in ("data", "examples", "dataset", "records")):
                    key = next(k for k in ("data", "examples", "dataset", "records") if k in data)
                    examples = data[key]
                    print(f"  ℹ️  Found {len(examples)} examples under key '{key}'")
                else:
                    load_errors.append(
                        "JSON object has no 'messages' key and no known wrapper key "
                        "(data/examples/dataset/records). Wrap your examples in an array."
                    )
        except json.JSONDecodeError as e:
            load_errors.append(f"JSON parse error: {e}")

    return examples, load_errors


def load_existing_hashes(path: str) -> set[str]:
    """Load fingerprints of already-existing examples to detect duplicates."""
    hashes = set()
    p = Path(path)
    if not p.exists():
        return hashes
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hashes.add(fingerprint(json.loads(line)))
    return hashes


def fingerprint(ex: dict) -> str:
    """Stable hash of an example's conversation content (ignores _meta)."""
    clean = {k: v for k, v in ex.items() if k != "_meta"}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict], output_path: Optional[str], dry_run: bool):
    total     = len(results)
    passed    = [r for r in results if r["valid"]]
    failed    = [r for r in results if not r["valid"]]
    duplicate = [r for r in results if r.get("duplicate")]
    written   = [r for r in results if r.get("written")]

    print("\n" + "═" * 55)
    print("  📊 IMPORT SUMMARY")
    print("═" * 55)
    print(f"  Total loaded     : {total}")
    print(f"  ✅ Valid         : {len(passed)}")
    print(f"  ❌ Invalid       : {len(failed)}")
    print(f"  ⚠️  Duplicates   : {len(duplicate)}")
    print(f"  💾 Written       : {len(written)}" + (" (dry run — nothing written)" if dry_run else ""))
    if output_path and not dry_run and written:
        print(f"  Output file      : {output_path}")
    print("═" * 55)

    if failed:
        print(f"\n  ❌ Failed examples ({len(failed)}):")
        for r in failed:
            print(f"\n  [{r['index']}] {r['preview']}")
            for err in r["errors"]:
                print(f"       • {err}")


def print_verbose(results: list[dict]):
    print("\n  📋 All results:")
    for r in results:
        icon = "✅" if r["valid"] else "❌"
        dup  = " (duplicate)" if r.get("duplicate") else ""
        wrt  = " → written" if r.get("written") else ""
        print(f"  [{r['index']:>3}] {icon}{dup}{wrt}  {r['preview']}")
        if not r["valid"]:
            for e in r["errors"]:
                print(f"         • {e}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import, validate, and append training examples to a JSONL dataset"
    )
    parser.add_argument("--input",    required=True,  help="Input .json or .jsonl file to import")
    parser.add_argument("--output",   default=None,   help="Output .jsonl file to append to")
    parser.add_argument("--dry-run",  action="store_true", help="Validate only — do not write anything")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate detection")
    parser.add_argument("--verbose",  action="store_true", help="Print result for every example")
    parser.add_argument("--strip-meta", action="store_true", help="Remove _meta keys before writing")
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        print("❌ --output is required unless using --dry-run")
        sys.exit(1)

    print("═" * 55)
    print("  🍛 Dataset Importer & Validator")
    print("═" * 55)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output or '(dry run)'}")
    print(f"  Dry run : {args.dry_run}")
    print(f"  Dedup   : {not args.no_dedup}")
    print("═" * 55)

    # ── Load input ──
    print(f"\n📂 Loading {args.input} ...")
    examples, load_errors = load_input_file(args.input)

    if load_errors:
        print(f"\n❌ File load errors:")
        for e in load_errors:
            print(f"   • {e}")
        if not examples:
            sys.exit(1)

    print(f"  Loaded {len(examples)} example(s)")

    # ── Load existing hashes for dedup ──
    existing_hashes: set[str] = set()
    if not args.no_dedup and args.output:
        existing_hashes = load_existing_hashes(args.output)
        if existing_hashes:
            print(f"  Found {len(existing_hashes)} existing example(s) in output file (dedup active)")

    # ── Validate + process each example ──
    print(f"\n🔍 Validating {len(examples)} example(s)...\n")

    results   = []
    to_write  = []

    for i, ex in enumerate(examples):
        # Short preview for error messages
        msgs = ex.get("messages", []) if isinstance(ex, dict) else []
        first_user = next((m.get("content","")[:60] for m in msgs if m.get("role") == "user"), f"example {i}")
        preview = first_user.replace("\n", " ")

        # ── Validate ──
        valid, errors = validate_example(ex)

        # ── Duplicate check ──
        is_dup = False
        if valid and not args.no_dedup:
            fp = fingerprint(ex)
            if fp in existing_hashes:
                is_dup = True
            else:
                existing_hashes.add(fp)

        # ── Decide whether to write ──
        should_write = valid and not is_dup and not args.dry_run

        if should_write:
            clean = {k: v for k, v in ex.items() if not (args.strip_meta and k == "_meta")}
            to_write.append(clean)

        results.append({
            "index":     i,
            "preview":   preview,
            "valid":     valid,
            "errors":    errors,
            "duplicate": is_dup,
            "written":   should_write,
        })

        # Live progress dot
        sym = "✅" if should_write else ("⚠️ " if is_dup else "❌")
        print(f"  [{i:>3}] {sym}  {preview[:55]}")
        if not valid and args.verbose:
            for e in errors:
                print(f"         • {e}")

    # ── Write to output ──
    if to_write and not args.dry_run:
        with open(args.output, "a", encoding="utf-8") as f:
            for ex in to_write:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── Summary ──
    if args.verbose:
        print_verbose(results)

    print_summary(results, args.output, args.dry_run)

    # ── Exit code: 0 if all valid, 1 if any failed ──
    failed_count = sum(1 for r in results if not r["valid"])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
    