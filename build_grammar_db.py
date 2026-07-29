#!/usr/bin/env python3
"""Build topik grammar SQLite (+ gzip) from topik_master_v1.json."""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path


def build_db(json_path: Path, db_path: Path) -> tuple[int, int]:
    items = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Expected a JSON array")

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE grammar (
                grammarId INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                englishMeaning TEXT,
                briefDescription TEXT,
                featuredExample TEXT,
                featuredTranslation TEXT,
                longerExplanation TEXT,
                level TEXT NOT NULL,
                sentenceExamples TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE grammar_similar (
                grammarId INTEGER NOT NULL,
                similarId INTEGER NOT NULL,
                PRIMARY KEY (grammarId, similarId),
                FOREIGN KEY (grammarId) REFERENCES grammar(grammarId),
                FOREIGN KEY (similarId) REFERENCES grammar(grammarId)
            )
            """
        )

        grammar_rows = []
        similar_rows = []
        for item in items:
            gid = item["grammarId"]
            grammar_rows.append(
                (
                    gid,
                    item.get("name"),
                    item.get("englishMeaning"),
                    item.get("briefDescription"),
                    item.get("featuredExample"),
                    item.get("featuredTranslation"),
                    item.get("longerExplanation"),
                    item.get("level"),
                    json.dumps(item.get("sentenceExamples") or [], ensure_ascii=False),
                )
            )
            for sid in item.get("similarToIds") or []:
                similar_rows.append((gid, sid))

        cur.executemany(
            """
            INSERT INTO grammar (
                grammarId, name, englishMeaning, briefDescription,
                featuredExample, featuredTranslation, longerExplanation,
                level, sentenceExamples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            grammar_rows,
        )
        cur.executemany(
            "INSERT INTO grammar_similar (grammarId, similarId) VALUES (?, ?)",
            similar_rows,
        )

        cur.execute("CREATE INDEX idx_grammar_level ON grammar(level)")
        cur.execute("CREATE INDEX idx_grammar_name ON grammar(name)")
        cur.execute("CREATE INDEX idx_grammar_similar_similarId ON grammar_similar(similarId)")

        conn.commit()
        conn.execute("VACUUM")
        return len(grammar_rows), len(similar_rows)
    finally:
        conn.close()


def compress_gzip(src: Path, dst: Path) -> None:
    with src.open("rb") as f_in, gzip.open(dst, "wb", compresslevel=9) as f_out:
        f_out.writelines(f_in)


def human_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TOPIK grammar SQLite from master JSON.")
    parser.add_argument("--input", "-i", default="topik_master_v1.json")
    parser.add_argument("--output-dir", "-o", default="db")
    parser.add_argument("--name", "-n", default="topik_grammar")
    parser.add_argument("--no-gzip", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.input)
    if not json_path.exists():
        parser.error(f"Input not found: {json_path}")

    out_dir = Path(args.output_dir)
    db_path = out_dir / f"{args.name}.sqlite"

    print(f"Building from {json_path} ...")
    n_grammar, n_similar = build_db(json_path, db_path)
    print(f"  grammar rows: {n_grammar}")
    print(f"  similar links: {n_similar}")
    print(f"  wrote {db_path} ({human_mb(db_path)})")

    if not args.no_gzip:
        gz_path = db_path.with_suffix(".sqlite.gz")
        compress_gzip(db_path, gz_path)
        print(f"  wrote {gz_path} ({human_mb(gz_path)})")

    print("Done.")


if __name__ == "__main__":
    main()
