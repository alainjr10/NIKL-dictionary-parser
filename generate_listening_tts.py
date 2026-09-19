#!/usr/bin/env python3
"""Generate TOPIK II listening drill audio via Gemini or OpenAI TTS APIs.

Default: pilot set (7 scripts). Use --all for the unique pool (~127).
Writes MP3 by default (WAV is only a temp file and is deleted after conversion).

Examples:
  # Gemini pilot (7) → drills_audio/topik2/listening/gemini/*.mp3
  ./.venv/Scripts/python.exe generate_listening_tts.py --provider gemini

  # Next 5 missing items from the unique pool (safe incremental batches)
  ./.venv/Scripts/python.exe generate_listening_tts.py --provider gemini --all --limit 5

  # Keep WAV intermediates as well
  ./.venv/Scripts/python.exe generate_listening_tts.py --provider gemini --all --limit 5 --keep-wav

  # One id only
  ./.venv/Scripts/python.exe generate_listening_tts.py --provider gemini --id ai2l_visual_match_001

  # Full unique pool (all ~127)
  ./.venv/Scripts/python.exe generate_listening_tts.py --provider gemini --all

Env:
  GEMINI_API_KEY   (gemini)
  OPENAI_API_KEY   (openai)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
PILOT_JSON = ROOT / "drills-prep/topik2/listening/tts_pilot.json"
UNIQUE_JSON = ROOT / "drills-prep/topik2/listening/tts_batch_unique.json"
DEFAULT_OUT = ROOT / "drills_audio/topik2/listening"

GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
OPENAI_MODEL = "gpt-4o-mini-tts"

# Gemini multi-speaker labels must be alphanumeric (no Hangul).
SPEAKER_MAP = {"남자": "Man", "여자": "Woman"}
GEMINI_VOICES = {"Man": "Charon", "Woman": "Kore"}
OPENAI_VOICES = {"Man": "onyx", "Woman": "nova"}

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
TURN_GAP_SEC = 0.45  # natural exam-style beat between speakers
BLANK_TAIL_SEC = 1.0

ROLE_PROMPT = """You are the official TOPIK II Listening exam audio narrator and cast director.

ROLE
- Generate ONLY spoken Korean exam audio for TOPIK II (한국어능력시험) Listening drills.
- You are not a chatbot, drama actor, or language tutor.
- Perform the transcript exactly in the style of real TOPIK listening recordings (NIIED).

CAST (fixed - never change)
- Man: adult Korean man, ~30-35, Seoul standard (표준어), calm, clear, mid pitch, informative and neutral. TOPIK exam male voice - not DJ, not cartoon.
- Woman: adult Korean woman, ~30-35, Seoul standard (표준어), calm, clear, mid pitch, firm and neutral. TOPIK exam female voice - not ASMR, not cute, not overly bright.

VOICE LOCK
- Use exactly Man and Woman as labeled in the transcript.
- Do not invent a third speaker.
- Do not speak speaker labels aloud.
- No music, SFX, reverb, laughter, whispering, or English.
- No question numbers, stems, options, answers, intros, or outros.
- Exact words only. Natural exam-studio pace.
- Numbers clearly in Korean (45% → 사십오 퍼센트).
"""

# OpenAI speech `instructions` — keep detailed; this strongly shapes Korean TOPIK timbre.
OPENAI_LINE_STYLE = {
    "Man": (
        "You are a native Korean man (~30–35) recording official TOPIK II "
        "(한국어능력시험) Listening exam audio for NIIED. "
        "Speak Seoul standard Korean (표준어) only — natural native pronunciation, "
        "rhythm, and intonation like real TOPIK dialogue recordings. "
        "Calm, clear, mid-to-low pitch, informative and neutral. "
        "Exam-studio quality: slightly clear enunciation, natural conversational "
        "pace (not rushed, not a slow lecture), mild politeness only. "
        "Not a radio DJ, YouTuber, cartoon, or English-accented AI assistant. "
        "No drama, whisper, laugh, music, or SFX. "
        "Read the Korean text exactly; add or omit nothing. Do not say speaker labels."
    ),
    "Woman": (
        "You are a native Korean woman (~30–35) recording official TOPIK II "
        "(한국어능력시험) Listening exam audio for NIIED. "
        "Speak Seoul standard Korean (표준어) only — natural native pronunciation, "
        "rhythm, and intonation like real TOPIK dialogue recordings. "
        "Calm, clear, mid pitch, firm and neutral. "
        "Exam-studio quality: slightly clear enunciation, natural conversational "
        "pace (not rushed, not a slow lecture), mild politeness only. "
        "Not ASMR, cute, bright customer-service AI, or English-accented. "
        "No drama, whisper, laugh, music, or SFX. "
        "Read the Korean text exactly; add or omit nothing. Do not say speaker labels."
    ),
}
# 1.0 = slightly brisker than many official TOPIK takes (good practice load).
OPENAI_SPEECH_SPEED = 1.0


def load_env() -> None:
    env_path = ROOT / ".env"
    if load_dotenv:
        load_dotenv(env_path)
        return
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def silence_pcm(seconds: float, rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


def wav_to_mp3(wav_path: Path) -> Path | None:
    if not shutil.which("ffmpeg"):
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        check=True,
        capture_output=True,
    )
    return mp3_path


def output_paths(out_dir: Path, item_id: str) -> tuple[Path, Path]:
    return out_dir / f"{item_id}.wav", out_dir / f"{item_id}.mp3"


def has_output(out_dir: Path, item_id: str, *, want_mp3: bool) -> bool:
    wav_path, mp3_path = output_paths(out_dir, item_id)
    if want_mp3:
        return mp3_path.exists()
    return wav_path.exists()


def parse_turns(script: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(남자|여자)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"Expected 남자:/여자: line, got: {line!r}")
        kr_label, text = m.group(1), m.group(2).strip()
        if not text or text == "( )":
            continue
        turns.append((SPEAKER_MAP[kr_label], text))
    if not turns:
        raise ValueError("No spoken turns in script")
    return turns


def extract_pcm(response: Any) -> bytes:
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None) or getattr(
                part, "inlineData", None
            )
            if not inline:
                continue
            data = getattr(inline, "data", None)
            if data:
                return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    raise RuntimeError("No audio data in TTS response.")


def write_wav_rate(path: Path, pcm: bytes, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def link_shared_audio_json(canonical_id: str, peer_ids: list[str]) -> None:
    """Point peer question rows at the canonical MP3 (no duplicate files).

    Updates filename on peers in tts_*.json / tts_scripts.json, keeps
    shared_groups.canonical_filename, and sets pool.json ``audio`` to the
    canonical basename for every id in the group.
    """
    canonical_file = f"{canonical_id}.mp3"
    group_ids = [canonical_id, *peer_ids]

    prep = ROOT / "drills-prep/topik2/listening"

    for name in ("tts_batch.json", "tts_batch_unique.json", "tts_pilot.json"):
        path = prep / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("items", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            rid = row.get("id")
            if rid in group_ids and "filename" in row:
                row["filename"] = canonical_file
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scripts_path = prep / "tts_scripts.json"
    if scripts_path.exists():
        data = json.loads(scripts_path.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            if item.get("id") in group_ids and "filename" in item:
                item["filename"] = canonical_file
        for g in data.get("shared_groups", []):
            qids = set(g.get("question_ids") or [])
            if canonical_id in qids or qids & set(group_ids):
                g["canonical_filename"] = canonical_file
        scripts_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    pool_path = prep / "pool.json"
    if pool_path.exists():
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        for row in pool:
            if row.get("id") in group_ids:
                row["audio"] = canonical_file
        pool_path.write_text(
            json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def synthesize_gemini_line(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    speaker: str,
    text: str,
    retries: int = 4,
) -> bytes:
    """One speaker line via Gemini single-voice TTS.

    Keep the prompt minimal: long role text often makes Flash-TTS emit
    text instead of audio (400 INVALID_ARGUMENT).
    """
    # Prefer bare transcript; fallback styles if the model misbehaves.
    prompts = [
        text,
        f"Read this Korean aloud exactly:\n{text}",
        (
            f"TTS only. Speak this Korean exactly, no other words. "
            f"{'Male' if speaker == 'Man' else 'Female'} TOPIK exam voice.\n{text}"
        ),
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        prompt = prompts[min(attempt, len(prompts) - 1)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types_mod.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types_mod.SpeechConfig(
                        voice_config=types_mod.VoiceConfig(
                            prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                                voice_name=GEMINI_VOICES[speaker],
                            )
                        )
                    ),
                ),
            )
            return extract_pcm(response)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if attempt < retries - 1 and (
                "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "tried to generate text" in msg
                or "INVALID_ARGUMENT" in msg
            ):
                wait = 8.0 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 2.0 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Gemini TTS failed after retries: {last_err}")


def synthesize_gemini(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    script: str,
    turn_gap_sec: float,
    blank_tail: bool,
    request_gap_sec: float = 6.5,
) -> tuple[bytes, int]:
    """Per-turn Gemini TTS, stitched with silence between speakers."""
    turns = parse_turns(script)
    pcm_parts: list[bytes] = []
    rate = SAMPLE_RATE
    for i, (speaker, text) in enumerate(turns):
        if i > 0 and request_gap_sec > 0:
            # Free-tier Flash TTS is often ~10 RPM; pace API calls.
            time.sleep(request_gap_sec)
        pcm_parts.append(
            synthesize_gemini_line(
                client, types_mod, model=model, speaker=speaker, text=text
            )
        )
        if i < len(turns) - 1 and turn_gap_sec > 0:
            pcm_parts.append(silence_pcm(turn_gap_sec, rate=rate))
    if blank_tail:
        pcm_parts.append(silence_pcm(BLANK_TAIL_SEC, rate=rate))
    return b"".join(pcm_parts), rate


def synthesize_openai(
    client: Any,
    *,
    model: str,
    script: str,
    turn_gap_sec: float,
    blank_tail: bool,
) -> tuple[bytes, int]:
    """One voice per turn, then stitch with silence (OpenAI has no native 2-speaker)."""
    turns = parse_turns(script)
    pcm_parts: list[bytes] = []
    rate = SAMPLE_RATE
    for i, (speaker, text) in enumerate(turns):
        voice = OPENAI_VOICES[speaker]
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions=OPENAI_LINE_STYLE[speaker],
            response_format="wav",
            speed=OPENAI_SPEECH_SPEED,
        )
        if hasattr(response, "read"):
            wav_bytes = response.read()
        elif hasattr(response, "content"):
            wav_bytes = response.content
        else:
            wav_bytes = bytes(response)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            if wf.getsampwidth() != SAMPLE_WIDTH or wf.getnchannels() != 1:
                raise RuntimeError(
                    f"Unexpected WAV format from OpenAI: "
                    f"ch={wf.getnchannels()} width={wf.getsampwidth()} rate={rate}"
                )
            frames = wf.readframes(wf.getnframes())
        pcm_parts.append(frames)
        if i < len(turns) - 1 and turn_gap_sec > 0:
            pcm_parts.append(silence_pcm(turn_gap_sec, rate=rate))
    if blank_tail:
        pcm_parts.append(silence_pcm(BLANK_TAIL_SEC, rate=rate))
    return b"".join(pcm_parts), rate


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all:
        path = UNIQUE_JSON
    else:
        path = Path(args.batch) if args.batch else PILOT_JSON
    items = json.loads(path.read_text(encoding="utf-8"))
    if args.id:
        wanted = set(args.id)
        items = [x for x in items if x["id"] in wanted]
        missing = wanted - {x["id"] for x in items}
        if missing:
            # allow --id from unique pool even in pilot mode
            all_items = json.loads(UNIQUE_JSON.read_text(encoding="utf-8"))
            by_id = {x["id"]: x for x in all_items}
            items = [by_id[i] for i in args.id if i in by_id]
            still = [i for i in args.id if i not in by_id]
            if still:
                sys.exit(f"Unknown id(s): {still}")
    return items


def select_items(
    items: list[dict[str, Any]],
    *,
    out_dir: Path,
    want_mp3: bool,
    force: bool,
    offset: int,
    limit: int | None,
    pending: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Apply offset/limit; optionally only missing outputs (pending).

    Returns (selected_items, pool_remaining_after_selection).
    """
    pool = items
    if offset:
        pool = pool[offset:]

    if pending and not force:
        pool = [
            item
            for item in pool
            if not has_output(out_dir, item["id"], want_mp3=want_mp3)
        ]

    remaining_after = 0
    if limit is not None:
        if limit < 1:
            sys.exit("--limit must be >= 1")
        remaining_after = max(0, len(pool) - limit)
        pool = pool[:limit]
    return pool, remaining_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TOPIK II listening drill TTS via Gemini or OpenAI API."
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        required=True,
        help="TTS provider",
    )
    parser.add_argument(
        "--batch",
        help="JSON list of {id, script, ...}. Default: tts_pilot.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use tts_batch_unique.json (all unique audio groups, ~127)",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="id",
        help="Only generate this id (repeatable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to generate this run (e.g. 5 or 10)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many items from the selected pool before limit",
    )
    parser.add_argument(
        "--pending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --limit/--all: only generate ids still missing output "
        "(default: true). Use --no-pending for a raw offset/limit slice.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Base output dir (provider subfolder added)",
    )
    parser.add_argument("--model", help="Override model id")
    parser.add_argument(
        "--mp3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert WAV → MP3 with ffmpeg (default: true). Use --no-mp3 for WAV only.",
    )
    parser.add_argument(
        "--keep-wav",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep WAV after MP3 conversion (default: false - MP3 only). Use --keep-wav to retain WAV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Pause between requests (seconds)",
    )
    parser.add_argument(
        "--turn-gap",
        type=float,
        default=TURN_GAP_SEC,
        help=f"Silence between speakers in seconds (default: {TURN_GAP_SEC})",
    )
    args = parser.parse_args()
    load_env()

    if args.offset < 0:
        sys.exit("--offset must be >= 0")
    if args.turn_gap < 0:
        sys.exit("--turn-gap must be >= 0")

    items = load_items(args)
    if not items:
        sys.exit("No items to generate.")

    out_dir = Path(args.out_dir) / args.provider
    out_dir.mkdir(parents=True, exist_ok=True)

    # Incremental batches: --all --limit 5 keeps picking the next missing ids.
    use_pending = args.pending and (args.limit is not None or args.all)
    items, remaining = select_items(
        items,
        out_dir=out_dir,
        want_mp3=args.mp3,
        force=args.force,
        offset=args.offset,
        limit=args.limit,
        pending=use_pending,
    )
    if not items:
        print("Nothing to generate (all selected outputs already exist).")
        print(f"Out: {out_dir}")
        return

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            sys.exit("Set GEMINI_API_KEY in .env or the environment.")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            sys.exit("Install google-genai: ./kr-dict-env/bin/pip install google-genai")
        client = genai.Client(api_key=api_key)
        model = args.model or GEMINI_MODEL
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            sys.exit("Set OPENAI_API_KEY in .env or the environment.")
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("Install openai: ./kr-dict-env/bin/pip install openai")
        client = OpenAI(api_key=api_key)
        model = args.model or OPENAI_MODEL

    if args.mp3 and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed for default MP3 output). Install ffmpeg or pass --no-mp3.")

    print(f"Provider: {args.provider}")
    print(f"Model:    {model}")
    print(f"Items:    {len(items)}" + (f"  (more pending after: {remaining})" if remaining else ""))
    print(f"Format:   {'mp3 (+wav)' if args.mp3 and args.keep_wav else 'mp3' if args.mp3 else 'wav'}")
    print(f"Turn gap: {args.turn_gap:.2f}s")
    print(f"Pending:  {use_pending}")
    print(f"Out:      {out_dir}")
    if args.provider == "gemini":
        print(f"Voices:   Man={GEMINI_VOICES['Man']} Woman={GEMINI_VOICES['Woman']} (per-turn stitch)")
    else:
        print(f"Voices:   Man={OPENAI_VOICES['Man']} Woman={OPENAI_VOICES['Woman']} (stitched)")

    ok = 0
    skipped = 0
    failed: list[str] = []

    for i, item in enumerate(items, 1):
        item_id = item["id"]
        wav_path, mp3_path = output_paths(out_dir, item_id)
        if has_output(out_dir, item_id, want_mp3=args.mp3) and not args.force:
            print(f"[{i}/{len(items)}] skip {item_id} (exists)")
            skipped += 1
            continue

        script = item["script"]
        blank_tail = "dialogue_continuation" in item_id or item_id.startswith(
            "ai2l_dialogue_continuation"
        )
        # also detect from skill if present
        if item.get("skill_type_id") == "dialogue_continuation":
            blank_tail = True

        print(f"[{i}/{len(items)}] {item_id} ...", flush=True)
        try:
            # WAV present but MP3 missing: convert only (no re-TTS).
            if (
                args.mp3
                and wav_path.exists()
                and not mp3_path.exists()
                and not args.force
            ):
                mp3 = wav_to_mp3(wav_path)
                if not mp3:
                    raise RuntimeError("ffmpeg MP3 conversion failed")
                extra = f" {mp3.name} (from existing wav)"
                if not args.keep_wav:
                    wav_path.unlink(missing_ok=True)
                peers = item.get("shared_with") or []
                if peers:
                    link_shared_audio_json(item_id, peers)
                    extra += f" links→{','.join(peers)}"
                print(f"  wrote{extra}")
                ok += 1
                if args.sleep > 0 and i < len(items):
                    time.sleep(args.sleep)
                continue

            if args.provider == "gemini":
                pcm, rate = synthesize_gemini(
                    client,
                    types,
                    model=model,
                    script=script,
                    turn_gap_sec=args.turn_gap,
                    blank_tail=blank_tail,
                )
            else:
                pcm, rate = synthesize_openai(
                    client,
                    model=model,
                    script=script,
                    turn_gap_sec=args.turn_gap,
                    blank_tail=blank_tail,
                )
            write_wav_rate(wav_path, pcm, rate=rate)
            extra = f" {wav_path.name}"
            if args.mp3:
                mp3 = wav_to_mp3(wav_path)
                if not mp3:
                    raise RuntimeError("ffmpeg MP3 conversion failed")
                extra = f" {mp3.name}"
                if not args.keep_wav:
                    wav_path.unlink(missing_ok=True)
            # One file per audio group — peers link via JSON (no duplicate copies).
            peers = item.get("shared_with") or []
            if peers:
                link_shared_audio_json(item_id, peers)
                extra += f" links→{','.join(peers)}"
            print(f"  wrote{extra}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 — surface per-item failures
            print(f"  FAILED: {exc}")
            failed.append(item_id)

        if args.sleep > 0 and i < len(items):
            time.sleep(args.sleep)

    print()
    print(f"Done. ok={ok} skipped={skipped} failed={len(failed)}")
    if remaining:
        print(f"Still pending after this batch: ~{remaining} (re-run with same --limit)")
    if failed:
        print("Failed ids:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
