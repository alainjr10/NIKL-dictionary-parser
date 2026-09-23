import json
from pathlib import Path

words = json.loads(
    Path("drills-prep/exams/topik1_91/audio/_whisper_words.json").read_text(encoding="utf-8")
)
needles = ["축하", "초등", "선수", "작년", "김수", "전국", "다음을", "물음"]
for w in words:
    if 2190 <= w["s"] <= 2320:
        if any(n in w["t"] for n in needles) or "번" in w["t"] or "다시" in w["t"]:
            print(f"{w['s']:7.1f}  {w['t']}")
