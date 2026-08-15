#!/usr/bin/env python3
"""Upsert TOPIK grammar from topik_master_v1.json into Supabase.

Prerequisites:
  1. Run sql/topik_grammar.sql in the Supabase SQL editor.
  2. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (service role, not anon).

Usage:
  python push_grammar_supabase.py
  python push_grammar_supabase.py -i topik_master_v1.json --replace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from supabase import create_client
except ImportError:
    sys.exit("Missing dependency. Install with: pip install supabase python-dotenv")


BATCH_SIZE = 200


def load_env() -> None:
    if load_dotenv:
        load_dotenv()


def to_row(item: dict) -> dict:
    examples = item.get("sentenceExamples") or []
    similar = item.get("similarToIds") or []
    return {
        "grammarId": item["grammarId"],
        "name": item["name"],
        "englishMeaning": item.get("englishMeaning"),
        "briefDescription": item.get("briefDescription"),
        "featuredExample": item.get("featuredExample"),
        "featuredTranslation": item.get("featuredTranslation"),
        "longerExplanation": item.get("longerExplanation"),
        "level": item["level"],
        "sentenceExamples": examples,
        "similarToIds": similar,
    }


def batched(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Push TOPIK grammar JSON to Supabase.")
    parser.add_argument("--input", "-i", default="topik_master_v1.json")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete all remote rows first, then insert (full refresh).",
    )
    args = parser.parse_args()

    load_env()
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        parser.error(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(copy .env.example to .env, or export the variables)."
        )

    json_path = Path(args.input)
    items = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        parser.error("Expected a JSON array")

    rows = [to_row(item) for item in items]
    client = create_client(url, key)
    table = client.table("topik_grammar")

    if args.replace:
        print("Clearing existing topik_grammar rows ...")
        table.delete().neq("grammarId", -1).execute()

    print(f"Upserting {len(rows)} grammar rows from {json_path} ...")
    uploaded = 0
    for chunk in batched(rows, BATCH_SIZE):
        table.upsert(chunk, on_conflict="grammarId").execute()
        uploaded += len(chunk)
        print(f"  {uploaded}/{len(rows)}")

    remote = table.select("grammarId", count="exact").execute()
    print(f"Done. Remote count: {remote.count}")


if __name__ == "__main__":
    main()
