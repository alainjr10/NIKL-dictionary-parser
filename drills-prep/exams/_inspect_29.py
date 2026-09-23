import json
from pathlib import Path

words = json.loads(
    Path("drills-prep/exams/topik1_91/audio/_whisper_words.json").read_text(encoding="utf-8")
)
lines = []
for w in words:
    if 2180 <= w["s"] <= 2330:
        lines.append(f"{w['s']:7.1f}  {w['t']}")
Path("drills-prep/exams/topik1_91/audio/_asr_29.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("\n".join(lines[:60]))
print("...")
print("\n".join(lines[-40:]))
