import json
from pathlib import Path

words = json.loads(
    Path("drills-prep/exams/topik1_91/audio/_whisper_words.json").read_text(encoding="utf-8")
)
for w in words:
    if 2170 <= w["s"] <= 2320 and (
        "김수미" in w["t"]
        or "수영" in w["t"]
        or "축하" in w["t"]
        or "다음을" in w["t"]
        or "다시" in w["t"]
        or w["t"] in {"27번", "28번", "29번", "30번"}
        or "번" in w["t"]
    ):
        print(f"{w['s']:7.1f}  {w['t']}")

print("--- full text 2195-2315 ---")
chunk = [w for w in words if 2195 <= w["s"] <= 2315]
print(" ".join(w["t"] for w in chunk))
