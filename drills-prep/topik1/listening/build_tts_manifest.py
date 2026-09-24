#!/usr/bin/env python3
"""Build TOPIK I listening TTS manifests from pool.json.

Writes:
  tts_scripts.json       — full inventory + shared_groups
  tts_batch_unique.json  — one row per unique audio (~153)
  tts_pilot.json         — small Gemini/OpenAI quality pilot

Usage:
  ./.venv/Scripts/python.exe drills-prep/topik1/listening/build_tts_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL_PATH = HERE / "pool.json"

STAGE_RE = re.compile(r"^\((?:딩동댕|댕동딩)\)\s*")
BLANK_RE = re.compile(r"^\(\s*\)$")

DELIVERY = {
    "listen_response": "response_prompt_blank",
    "listen_followup": "followup_blank",
    "listen_location": "short_everyday_dialogue",
    "listen_topic": "short_everyday_dialogue",
    "listen_picture": "short_everyday_dialogue",
    "listen_content_match": "short_everyday_dialogue",
    "listen_main_idea": "main_idea_monologue_or_dialogue",
    "listen_public_announcement": "public_announcement_broadcast",
    "listen_conversation_detail": "paired_conversation",
    "listen_interview": "paired_interview",
}

PILOT_IDS = [
    "ai1l_resp_q1_001",
    "ai1l_fol_q5_001",
    "ai1l_loc_q7_001",
    "ai1l_pic_q15_001",
    "ai1l_cm_q17_001",
    "ai1l_mi_q22_001",
    "ai1l_ann_q25_001",
    "ai1l_conv_q27_001",
]


def clean_script(transcript: str) -> str:
    lines: list[str] = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(남자|여자)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"Expected 남자:/여자: line, got: {line!r}")
        label, text = m.group(1), m.group(2).strip()
        text = STAGE_RE.sub("", text).strip()
        if BLANK_RE.match(text) or text == "( )":
            lines.append(f"{label}: ( )")
            continue
        if not text:
            continue
        lines.append(f"{label}: {text}")
    if not lines:
        raise ValueError("Empty script after cleaning")
    return "\n".join(lines)


def audio_group_id(script: str) -> str:
    return "ag_" + hashlib.sha1(script.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    scripts = {q["id"]: clean_script(q["transcript"]) for q in pool}

    # Validate passage groups share one script.
    by_group: dict[str, set[str]] = {}
    for q in pool:
        gid = q.get("passage_group_id")
        if not gid:
            continue
        by_group.setdefault(gid, set()).add(scripts[q["id"]])
    for gid, variants in by_group.items():
        if len(variants) > 1:
            raise SystemExit(f"passage_group script mismatch: {gid}")

    items_all: list[dict] = []
    unique: list[dict] = []
    shared_groups: list[dict] = []
    seen_ag: set[str] = set()

    for q in pool:
        script = scripts[q["id"]]
        ag = audio_group_id(script)
        if q.get("passage_group_id"):
            peers = [
                x["id"]
                for x in pool
                if x.get("passage_group_id") == q["passage_group_id"]
                and x["id"] != q["id"]
            ]
        else:
            peers = [
                x["id"]
                for x in pool
                if x["id"] != q["id"] and scripts[x["id"]] == script
            ]

        row = {
            "id": q["id"],
            "audio_group_id": ag,
            "filename": f"{q['id']}.mp3",
            "shared_with": peers,
            "delivery": DELIVERY.get(q["skill_type_id"], "dialogue"),
            "skill_type_id": q["skill_type_id"],
            "model_q_num": q.get("model_q_num"),
            "script": script,
            "tts_script": script,
            "topic": q.get("topic"),
        }
        items_all.append(row)

        if ag in seen_ag:
            continue
        seen_ag.add(ag)
        unique.append(
            {
                "id": q["id"],
                "audio_group_id": ag,
                "filename": f"{q['id']}.mp3",
                "shared_with": peers,
                "delivery": row["delivery"],
                "skill_type_id": q["skill_type_id"],
                "model_q_num": q.get("model_q_num"),
                "script": script,
            }
        )
        if peers:
            shared_groups.append(
                {
                    "audio_group_id": ag,
                    "question_ids": [q["id"], *peers],
                    "canonical_filename": f"{q['id']}.mp3",
                }
            )

    by_unique = {x["id"]: x for x in unique}
    missing = [i for i in PILOT_IDS if i not in by_unique]
    if missing:
        raise SystemExit(f"Pilot ids missing from unique pool: {missing}")
    pilot = [by_unique[i] for i in PILOT_IDS]

    scripts_doc = {
        "level": "topik1",
        "section": "listening",
        "count": len(items_all),
        "unique_audio_count": len(unique),
        "audio_folder": "drills_audio/topik1/listening",
        "naming": "{id}.mp3",
        "shared_groups": shared_groups,
        "items": items_all,
    }

    (HERE / "tts_scripts.json").write_text(
        json.dumps(scripts_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "tts_batch_unique.json").write_text(
        json.dumps(unique, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "tts_pilot.json").write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Wrote manifests: items={len(items_all)} unique={len(unique)} "
        f"shared_groups={len(shared_groups)} pilot={len(pilot)}"
    )


if __name__ == "__main__":
    main()
