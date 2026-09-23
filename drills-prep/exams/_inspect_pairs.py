import json
import re
from pathlib import Path

words = json.loads(
    Path("drills-prep/exams/topik1_91/audio/_whisper_words.json").read_text(encoding="utf-8")
)

# Find section cues and "다시 들으십시오"
cues = []
for i, w in enumerate(words):
    t = w["t"]
    window = "".join(x["t"] for x in words[i : i + 6])
    if "다시" in t and i + 1 < len(words) and "들으" in words[i + 1]["t"]:
        cues.append(("replay", w["s"], window[:40]))
    if t.startswith("다음을") or (t == "다음을"):
        cues.append(("next_listen", w["s"], window[:50]))
    if "번입니다" in t:
        cues.append(("section", w["s"], t))

print("Cues after 1700s:")
for kind, s, w in cues:
    if s >= 1700:
        print(f"  {s:7.1f}  {kind:12}  {w}")

# Distinctive starts for pair passages
needles = [
    "주말에도",
    "인주마트",
    "인주",
    "종이",
    "미술관",
    "김수미",
    "수영",
    "전국",
]
print("\nNeedle hits:")
for needle in needles:
    for w in words:
        if needle in w["t"] and w["s"] >= 1700:
            print(f"  {w['s']:7.1f}  {needle}  ({w['t']})")
