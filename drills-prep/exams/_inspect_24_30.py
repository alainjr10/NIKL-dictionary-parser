import json
import re
from pathlib import Path

words = json.loads(
    Path("drills-prep/exams/topik1_91/audio/_whisper_words.json").read_text(encoding="utf-8")
)
listening = json.loads(
    Path("drills-prep/exams/topik1_91/listening.json").read_text(encoding="utf-8")
)

# Dump 24-end region with readable UTF-8
lines = []
for w in words:
    if 1750 <= w["s"] <= 2450:
        lines.append(f"{w['s']:7.1f}  {w['e']:7.1f}  {w['t']}")
Path("drills-prep/exams/topik1_91/audio/_asr_24_30.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("wrote", len(lines), "lines")

# Search for distinctive Q25 keywords
keys = ["인주마트", "마트", "채소", "아홉", "열 시", "주말", "종이", "미술관", "수영", "김수미"]
for key in keys:
    hits = [w for w in words if key in w["t"]]
    if hits:
        print(key, "->", [(round(h["s"], 1), h["t"]) for h in hits[:8]])

# All 번-like tokens 1700+
print("\n번 tokens:")
for w in words:
    if w["s"] < 1700:
        continue
    if "번" in w["t"]:
        print(f"{w['s']:.1f}\t{w['t']}")
