#!/usr/bin/env python3
"""Generate TOPIK II listening drill audio via Gemini or OpenAI TTS APIs.

Default: pilot set (7 scripts). Use --all for every unique audio group.

Examples:
  # Gemini pilot → drills_audio/topik2/listening/gemini/*.wav
  ./kr-dict-env/bin/python generate_listening_tts.py --provider gemini

  # OpenAI pilot → drills_audio/topik2/listening/openai/*.wav
  ./kr-dict-env/bin/python generate_listening_tts.py --provider openai

  # One id only
  ./kr-dict-env/bin/python generate_listening_tts.py --provider gemini --id ai2l_visual_match_001

  # Full unique pool
  ./kr-dict-env/bin/python generate_listening_tts.py --provider gemini --all

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
TURN_GAP_SEC = 0.30
BLANK_TAIL_SEC = 1.0

ROLE_PROMPT = """You are the official TOPIK II Listening exam audio narrator and cast director.

ROLE
- Generate ONLY spoken Korean exam audio for TOPIK II (한국어능력시험) Listening drills.
- You are not a chatbot, drama actor, or language tutor.
- Perform the transcript exactly in the style of real TOPIK listening recordings (NIIED).

CAST (fixed — never change)
- Man: adult Korean man, ~30–35, Seoul standard (표준어), calm, clear, mid pitch, informative and neutral. TOPIK exam male voice — not DJ, not cartoon.
- Woman: adult Korean woman, ~30–35, Seoul standard (표준어), calm, clear, mid pitch, firm and neutral. TOPIK exam female voice — not ASMR, not cute, not overly bright.

VOICE LOCK
- Use exactly Man and Woman as labeled in the transcript.
- Do not invent a third speaker.
- Do not speak speaker labels aloud.
- No music, SFX, reverb, laughter, whispering, or English.
- No question numbers, stems, options, answers, intros, or outros.
- Exact words only. Natural exam-studio pace. Short pause between turns.
- Numbers clearly in Korean (45% → 사십오 퍼센트).
"""

OPENAI_LINE_INSTRUCTIONS = (
    "Speak as a TOPIK II Korean listening-exam voice. Seoul standard Korean, "
    "calm, clear, neutral, exam-studio quality. Natural conversational pace, "
    "slightly clear. No English, no drama, no music, no extra words. "
    "Recite the text exactly."
)


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


def silence_pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


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


def build_gemini_prompt(script: str, *, delivery: str, item_id: str) -> str:
    turns = parse_turns(script)
    dialogue = "\n".join(f"{spk}: {text}" for spk, text in turns)
    return "\n".join(
        [
            ROLE_PROMPT.strip(),
            "",
            "Synthesize this multi-speaker TOPIK II listening conversation as continuous audio.",
            "Do not answer in text — audio only.",
            f"Item id: {item_id}",
            f"Delivery target: {delivery}",
            "",
            "TRANSCRIPT",
            dialogue,
        ]
    )


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


def synthesize_gemini(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    prompt: str,
) -> bytes:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_mod.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types_mod.SpeechConfig(
                multi_speaker_voice_config=types_mod.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types_mod.SpeakerVoiceConfig(
                            speaker="Man",
                            voice_config=types_mod.VoiceConfig(
                                prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                                    voice_name=GEMINI_VOICES["Man"],
                                )
                            ),
                        ),
                        types_mod.SpeakerVoiceConfig(
                            speaker="Woman",
                            voice_config=types_mod.VoiceConfig(
                                prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                                    voice_name=GEMINI_VOICES["Woman"],
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    )
    return extract_pcm(response)


def synthesize_openai(
    client: Any,
    *,
    model: str,
    script: str,
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
            instructions=OPENAI_LINE_INSTRUCTIONS,
            response_format="wav",
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
        if i < len(turns) - 1:
            pcm_parts.append(b"\x00\x00" * int(rate * TURN_GAP_SEC))
    if blank_tail:
        pcm_parts.append(b"\x00\x00" * int(rate * BLANK_TAIL_SEC))
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
        help="Use tts_batch_unique.json (all unique audio groups)",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="id",
        help="Only generate this id (repeatable)",
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
        action="store_true",
        help="Also convert WAV → MP3 with ffmpeg if available",
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
    args = parser.parse_args()
    load_env()

    items = load_items(args)
    if not items:
        sys.exit("No items to generate.")

    out_dir = Path(args.out_dir) / args.provider
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Provider: {args.provider}")
    print(f"Model:    {model}")
    print(f"Items:    {len(items)}")
    print(f"Out:      {out_dir}")
    if args.provider == "gemini":
        print(f"Voices:   Man={GEMINI_VOICES['Man']} Woman={GEMINI_VOICES['Woman']}")
    else:
        print(f"Voices:   Man={OPENAI_VOICES['Man']} Woman={OPENAI_VOICES['Woman']} (stitched)")

    ok = 0
    skipped = 0
    failed: list[str] = []

    for i, item in enumerate(items, 1):
        item_id = item["id"]
        wav_path = out_dir / f"{item_id}.wav"
        if wav_path.exists() and not args.force:
            print(f"[{i}/{len(items)}] skip {item_id} (exists)")
            skipped += 1
            continue

        script = item["script"]
        delivery = item.get("delivery", "")
        blank_tail = "dialogue_continuation" in item_id or item_id.startswith(
            "ai2l_dialogue_continuation"
        )
        # also detect from skill if present
        if item.get("skill_type_id") == "dialogue_continuation":
            blank_tail = True

        print(f"[{i}/{len(items)}] {item_id} ...", flush=True)
        try:
            if args.provider == "gemini":
                prompt = build_gemini_prompt(
                    script, delivery=delivery, item_id=item_id
                )
                pcm = synthesize_gemini(client, types, model=model, prompt=prompt)
                if blank_tail:
                    pcm = pcm + silence_pcm(BLANK_TAIL_SEC)
                rate = SAMPLE_RATE
            else:
                pcm, rate = synthesize_openai(
                    client, model=model, script=script, blank_tail=blank_tail
                )
            write_wav_rate(wav_path, pcm, rate=rate)
            extra = ""
            if args.mp3:
                mp3 = wav_to_mp3(wav_path)
                extra = f" + {mp3.name}" if mp3 else " (ffmpeg missing, wav only)"
            for peer in item.get("shared_with") or []:
                peer_path = out_dir / f"{peer}.wav"
                if args.force or not peer_path.exists():
                    shutil.copy2(wav_path, peer_path)
                    extra += f" alias→{peer}"
            print(f"  wrote {wav_path.name}{extra}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 — surface per-item failures
            print(f"  FAILED: {exc}")
            failed.append(item_id)

        if args.sleep > 0 and i < len(items):
            time.sleep(args.sleep)

    print()
    print(f"Done. ok={ok} skipped={skipped} failed={len(failed)}")
    if failed:
        print("Failed ids:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
