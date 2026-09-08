#!/usr/bin/env python3
"""Generate Gemini 2.5 Flash TTS audio for stories.

Modes:
  (default) Per-line clips under out/story-audio/ST_001/line_00.mp3
  --full    One coherent dialogue/narration WAV at story_audio/ST_001.wav
            (then split with split_story_audio.py)

Multi-speaker full mode uses Gemini multiSpeakerVoiceConfig (max 2 speakers).
Narrator/essay stories use a single voice for the whole block.

Prerequisites:
  pip install google-genai python-dotenv supabase
  Optional for MP3: ffmpeg on PATH (otherwise writes .wav)

Env:
  GEMINI_API_KEY                  (required)
  SUPABASE_URL                    (required with --upload)
  SUPABASE_SERVICE_ROLE_KEY       (required with --upload)

Examples:
  python generate_story_audio.py ST_007 ST_008 ST_009 ST_010 --full \\
    --out-dir story_audio \\
    --tone "natural intermediate Korean; clear enunciation"

  python generate_story_audio.py ST_001 \\
    --tone "polite workplace Korean; manager firm but calm; employee apologetic" \\
    --context "Office apology after a missed deadline; natural conversational pace"
"""
from __future__ import annotations

import argparse
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

DEFAULT_STORIES = Path(__file__).parent / "res" / "stories_content.json"
DEFAULT_OUT = Path(__file__).parent / "out" / "story-audio"
DEFAULT_FULL_OUT = Path(__file__).parent / "story_audio"
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Prebuilt Gemini TTS voices — rotated per speaker within a story.
VOICE_POOL = [
    "Kore",
    "Puck",
    "Aoede",
    "Charon",
    "Fenrir",
    "Leda",
    "Orus",
    "Zephyr",
]

# Gemini multi-speaker aliases must be alphanumeric (no Hangul/whitespace).
SPEAKER_ALIAS_HINTS = {
    "면접관": "Interviewer",
    "지원자": "Candidate",
    "지수": "Jisu",
    "마크": "Mark",
    "민호": "Minho",
    "서연": "Seoyeon",
    "민지": "Minji",
    "톰": "Tom",
    "미나": "Mina",
    "준": "Jun",
    "선생님": "Teacher",
    "수진": "Sujin",
    "직원": "Clerk",
    "손님": "Customer",
    "유진": "Yujin",
    "현우": "Hyunwoo",
    "지민": "Jimin",
    "민수": "Minsu",
    "수미": "Sumi",
    "준호": "Junho",
    "하늘": "Haneul",
    "지안": "Jian",
    "지후": "Jihu",
    "에밀리": "Emily",
    "환자": "Patient",
    "한의사": "Doctor",
    "앵커": "Anchor",
    "기자": "Reporter",
    "전문가": "Expert",
    "Narrator": "Narrator",
}


def load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if load_dotenv:
        load_dotenv(env_path)
        return
    # Minimal .env loader when python-dotenv is not installed.
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


def load_story(stories_path: Path, story_id: str) -> dict[str, Any]:
    data = json.loads(stories_path.read_text(encoding="utf-8"))
    for story in data:
        if story.get("id") == story_id:
            return story
    known = ", ".join(s.get("id", "?") for s in data[:8])
    raise SystemExit(f"Story {story_id!r} not found. Examples: {known}...")


def story_speakers(story: dict[str, Any]) -> list[str]:
    speakers: list[str] = []
    for line in story.get("lines", []):
        sp = line.get("speaker") or "Narrator"
        if sp not in speakers:
            speakers.append(sp)
    return speakers or ["Narrator"]


def assign_voices(
    story: dict[str, Any],
    voice_map: dict[str, str] | None,
    default_voice: str,
) -> dict[str, str]:
    """Map each speaker (or Narrator for non-dialogue) to a stable voice."""
    mapping = dict(voice_map or {})
    speakers = story_speakers(story)

    used = set(mapping.values())
    pool = [v for v in VOICE_POOL if v not in used]
    if default_voice in VOICE_POOL:
        # Prefer requested default for first unmapped speaker
        pool = [default_voice] + [v for v in pool if v != default_voice]

    for i, sp in enumerate(speakers):
        if sp not in mapping:
            mapping[sp] = pool[i % len(pool)] if pool else default_voice
    # Keep only speakers that appear in this story (ignore extra --voice-map keys).
    return {sp: mapping[sp] for sp in speakers}


def speaker_aliases(speakers: list[str]) -> dict[str, str]:
    """Map story speaker labels -> alphanumeric aliases for multi-speaker TTS."""
    used: set[str] = set()
    aliases: dict[str, str] = {}
    for i, sp in enumerate(speakers):
        hint = SPEAKER_ALIAS_HINTS.get(sp)
        if hint and re.fullmatch(r"[A-Za-z0-9]+", hint) and hint not in used:
            alias = hint
        elif re.fullmatch(r"[A-Za-z0-9]+", sp) and sp not in used:
            alias = sp
        else:
            alias = f"Speaker{i + 1}"
            n = i + 1
            while alias in used:
                n += 1
                alias = f"Speaker{n}"
        aliases[sp] = alias
        used.add(alias)
    return aliases


def parse_kv_map(raw: str | None, flag: str, sep: str = ",") -> dict[str, str]:
    """Parse 'Name=value' maps. Use sep between entries (',' or '|' for long personas)."""
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for chunk in raw.split(sep):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(
                f"Invalid {flag} entry {chunk!r}; use Name=value "
                f"(separate entries with {sep!r})"
            )
        name, value = chunk.split("=", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def default_persona(speaker: str, story_type: str) -> str:
    """Stable identity description to reduce voice drift across line calls."""
    if speaker == "Narrator":
        if story_type in {"news", "announcement"}:
            return (
                "Korean news/announcer voice, mid adult, clear and neutral, "
                "steady mid pitch, professional broadcast delivery, same person every line"
            )
        return (
            "Korean story narrator, warm adult voice, calm mid pitch, "
            "even pacing, same person every line"
        )
    # Conversation roles - keep identity fixed; mood can shift slightly but not timbre
    if any(x in speaker for x in ("팀장", "과장", "부장", "사장님", "선생님", "원장", "면접관", "한의사")):
        return (
            f"{speaker}: Korean adult professional, slightly lower mid pitch, "
            "composed and mature, steady timbre, same person every line"
        )
    if "지원자" in speaker or "환자" in speaker:
        return (
            f"{speaker}: Korean young adult, clear polite mid pitch, "
            "respectful and natural, same person every line"
        )
    if any(x in speaker for x in ("앵커", "기자", "전문가", "리포터", "캐스터")):
        return (
            f"{speaker}: Korean broadcast voice, clear diction, neutral mid pitch, "
            "same person every line"
        )
    return (
        f"{speaker}: Korean young adult, natural conversational mid pitch, "
        "consistent timbre and age, same person every line"
    )


def build_prompt(
    *,
    text: str,
    speaker: str,
    voice: str,
    story: dict[str, Any],
    tone: str,
    context: str,
    persona: str,
    line_index: int,
    total_lines: int,
) -> str:
    """Structured TTS prompt (profile/scene/notes + TRANSCRIPT delimiter)."""
    title = story.get("title", story["id"])
    story_type = story.get("type", "story")
    intro = story.get("intro", "")

    scene_bits = [f"Language-learning story audio for '{title}' ({story_type})."]
    if intro:
        scene_bits.append(intro)
    if context:
        scene_bits.append(context)
    scene_bits.append(
        f"This is line {line_index + 1} of {total_lines} from the same scene. "
        "Keep continuity with the same cast and setting."
    )

    style_bits = [
        "Native Korean speech only.",
        "Recite the transcript EXACTLY - never add, omit, translate, or read stage directions aloud.",
        "Keep this speaker's vocal identity fixed: same gender presentation, age, pitch range, and timbre as previous lines.",
        "Do not reinvent the character or switch to a different-sounding person.",
        "Natural conversational Korean pacing for intermediate learners; clear enunciation.",
    ]
    if tone:
        style_bits.append(f"Scene tone guidance: {tone}")
    style_bits.append(
        "Emotion may fit the line, but must stay within this locked voice identity."
    )

    return "\n".join(
        [
            "Synthesize speech audio from the transcript below. Do not answer in text.",
            "",
            f"# AUDIO PROFILE: {speaker}",
            f"## Voice preset: {voice}",
            f"## Character lock: {persona}",
            "",
            "## THE SCENE",
            " ".join(scene_bits),
            "",
            "### DIRECTOR'S NOTES",
            f"Style: {' '.join(style_bits)}",
            "Pace: Natural spoken Korean, moderate speed, slight pause at commas/periods.",
            "Accent: Standard Seoul Korean.",
            "Consistency: Same speaker as all other lines for this character in this story.",
            "",
            "#### TRANSCRIPT",
            text,
        ]
    )


def pcm_to_wav(pcm: bytes, wav_path: Path) -> None:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg, or pass --format wav."
        )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-800:]}")


def extract_pcm(response: Any) -> bytes:
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected TTS response shape: {response!r}") from exc

    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if data:
            return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    raise RuntimeError("No audio data in TTS response.")


def get_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Missing dependency. Install with: pip install google-genai")
    return genai, types


def synthesize_line(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    prompt: str,
    voice: str,
) -> bytes:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_mod.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types_mod.SpeechConfig(
                voice_config=types_mod.VoiceConfig(
                    prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )
    return extract_pcm(response)


def chunk_lines_for_tts(
    lines: list[dict[str, Any]],
    max_speakers: int = 2,
) -> list[list[dict[str, Any]]]:
    """Split lines into chunks with at most max_speakers distinct speakers each.

    Gemini multi-speaker TTS allows only 2 voices per request; news casts with
    Anchor/Reporter/Expert are stitched from successive 2-speaker chunks.
    """
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    active: set[str] = set()
    for line in lines:
        if not (line.get("kr") or "").strip():
            continue
        sp = line.get("speaker") or "Narrator"
        if sp not in active and len(active) >= max_speakers:
            chunks.append(current)
            current = [line]
            active = {sp}
        else:
            current.append(line)
            active.add(sp)
    if current:
        chunks.append(current)
    return chunks


def silence_pcm(seconds: float = 0.25) -> bytes:
    n_frames = int(SAMPLE_RATE * seconds)
    return b"\x00\x00" * n_frames


def build_full_prompt(
    *,
    story: dict[str, Any],
    tone: str,
    context: str,
    persona_map: dict[str, str],
    alias_map: dict[str, str],
    multi: bool,
    lines: list[dict[str, Any]] | None = None,
    chunk_note: str = "",
) -> str:
    """One prompt for a story (or a 2-speaker chunk of a larger cast)."""
    title = story.get("title", story["id"])
    story_type = story.get("type", "story")
    intro = story.get("intro", "")
    lines = list(lines if lines is not None else (story.get("lines") or []))

    scene_bits = [f"Language-learning story audio for '{title}' ({story_type})."]
    if intro:
        scene_bits.append(intro)
    if context:
        scene_bits.append(context)
    if chunk_note:
        scene_bits.append(chunk_note)

    style_bits = [
        "Native Korean speech only.",
        "Recite the transcript EXACTLY — never add, omit, translate, or read stage directions aloud.",
        "Do not speak speaker labels aloud; use them only to switch voices between turns.",
        "Natural conversational Korean pacing for intermediate learners; clear enunciation.",
        "Keep each speaker's vocal identity fixed across the whole dialogue.",
    ]
    if tone:
        style_bits.append(f"Scene tone guidance: {tone}")

    cast_lines = []
    for sp, alias in alias_map.items():
        cast_lines.append(f"- {alias} ({sp}): {persona_map.get(sp, '')}")

    if multi:
        dialogue_lines = []
        for line in lines:
            text = (line.get("kr") or "").strip()
            if not text:
                continue
            sp = line.get("speaker") or "Narrator"
            dialogue_lines.append(f"{alias_map[sp]}: {text}")
        transcript = "\n".join(dialogue_lines)
        header = "Synthesize this multi-speaker conversation as continuous audio."
    else:
        paragraphs = [(line.get("kr") or "").strip() for line in lines]
        transcript = "\n".join(p for p in paragraphs if p)
        header = "Synthesize this narration as continuous single-speaker audio."

    return "\n".join(
        [
            header,
            "Do not answer in text — audio only.",
            "",
            "## THE SCENE",
            " ".join(scene_bits),
            "",
            "## CAST",
            *cast_lines,
            "",
            "### DIRECTOR'S NOTES",
            f"Style: {' '.join(style_bits)}",
            "Pace: Natural spoken Korean, moderate speed, slight pause at commas/periods and between turns.",
            "Accent: Standard Seoul Korean.",
            "",
            "#### TRANSCRIPT",
            transcript,
        ]
    )


def synthesize_full(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    prompt: str,
    voice_map: dict[str, str],
    alias_map: dict[str, str],
) -> bytes:
    speakers = list(alias_map.keys())
    if len(speakers) > 2:
        raise RuntimeError(
            f"synthesize_full expected <=2 speakers, got {speakers}"
        )
    if len(speakers) == 2:
        speaker_cfgs = []
        for sp in speakers:
            speaker_cfgs.append(
                types_mod.SpeakerVoiceConfig(
                    speaker=alias_map[sp],
                    voice_config=types_mod.VoiceConfig(
                        prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                            voice_name=voice_map[sp],
                        )
                    ),
                )
            )
        speech_config = types_mod.SpeechConfig(
            multi_speaker_voice_config=types_mod.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_cfgs,
            )
        )
    else:
        sp = speakers[0]
        speech_config = types_mod.SpeechConfig(
            voice_config=types_mod.VoiceConfig(
                prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                    voice_name=voice_map[sp],
                )
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_mod.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        ),
    )
    return extract_pcm(response)


def upload_folder(
    local_dir: Path,
    *,
    bucket: str,
    story_id: str,
    content_type: str,
) -> list[str]:
    try:
        from supabase import create_client
    except ImportError:
        sys.exit("Missing dependency. Install with: pip install supabase")

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise SystemExit(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for --upload."
        )

    sb = create_client(url, key)
    uploaded: list[str] = []
    for path in sorted(local_dir.iterdir()):
        if path.suffix.lower() not in {".mp3", ".wav", ".json"}:
            continue
        remote = f"{story_id}/{path.name}"
        data = path.read_bytes()
        ctype = "application/json" if path.suffix == ".json" else content_type
        # Upsert: remove then upload (compatible across supabase-py versions)
        try:
            sb.storage.from_(bucket).remove([remote])
        except Exception:
            pass
        sb.storage.from_(bucket).upload(
            remote,
            data,
            {"content-type": ctype, "upsert": "true"},
        )
        uploaded.append(remote)
    return uploaded


def parse_voice_map(raw: str | None) -> dict[str, str]:
    return parse_kv_map(raw, "--voice-map")


def assign_personas(
    story: dict[str, Any],
    persona_map: dict[str, str],
) -> dict[str, str]:
    story_type = story.get("type", "story")
    mapping = dict(persona_map)
    for line in story.get("lines", []):
        sp = line.get("speaker") or "Narrator"
        if sp not in mapping:
            mapping[sp] = default_persona(sp, story_type)
    return mapping


def prompt_if_missing(value: str | None, label: str, *, interactive: bool) -> str:
    if value is not None:
        return value.strip()
    if not interactive:
        return ""
    try:
        entered = input(f"{label} (optional, Enter to skip): ").strip()
    except EOFError:
        entered = ""
    return entered


def generate_full_story(
    *,
    story: dict[str, Any],
    out_dir: Path,
    tone: str,
    context: str,
    voice_map: dict[str, str],
    persona_map: dict[str, str],
    model: str,
    dry_run: bool,
    skip_existing: bool,
    client: Any,
    types_mod: Any,
    sleep_s: float = 0.5,
) -> Path:
    """Write story_audio/ST_00X.wav (full dialogue/narration; stitch if >2 speakers)."""
    story_id = story["id"]
    speakers = story_speakers(story)
    full_alias_map = speaker_aliases(speakers)
    lines = [ln for ln in (story.get("lines") or []) if (ln.get("kr") or "").strip()]
    chunks = chunk_lines_for_tts(lines, max_speakers=2)

    out_path = out_dir / f"{story_id}.wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = (
        "single-speaker"
        if len(speakers) == 1
        else ("multi-speaker" if len(chunks) == 1 else f"stitched multi ({len(chunks)} chunks)")
    )
    print(f"Story: {story_id} - {story.get('title')}")
    print(f"Mode:  full {mode}")
    print("Voices / personas:")
    for sp, voice in voice_map.items():
        print(f"  {full_alias_map[sp]} ({sp}) -> {voice}")
        print(f"      {persona_map.get(sp, '')}")
    if tone:
        print(f"Tone: {tone}")
    if context:
        print(f"Context: {context}")
    print(f"Output: {out_path}")

    if skip_existing and out_path.exists():
        print(f"  exists — skipping {out_path.name}")
        return out_path

    pcm_parts: list[bytes] = []
    for ci, chunk in enumerate(chunks):
        chunk_speakers: list[str] = []
        for line in chunk:
            sp = line.get("speaker") or "Narrator"
            if sp not in chunk_speakers:
                chunk_speakers.append(sp)
        chunk_aliases = {sp: full_alias_map[sp] for sp in chunk_speakers}
        chunk_voices = {sp: voice_map[sp] for sp in chunk_speakers}
        multi = len(chunk_speakers) >= 2
        chunk_note = ""
        if len(chunks) > 1:
            chunk_note = (
                f"This is segment {ci + 1} of {len(chunks)} from the same continuous "
                "broadcast/scene; match prior segment energy and keep identities locked."
            )
        prompt = build_full_prompt(
            story=story,
            tone=tone,
            context=context,
            persona_map=persona_map,
            alias_map=chunk_aliases,
            multi=multi,
            lines=chunk,
            chunk_note=chunk_note,
        )

        if dry_run:
            print(f"\n--- full prompt chunk {ci + 1}/{len(chunks)} ---")
            print(prompt)
            continue

        label = "/".join(chunk_speakers)
        print(
            f"  synthesizing chunk {ci + 1}/{len(chunks)} ({label})...",
            flush=True,
        )
        pcm = synthesize_full(
            client,
            types_mod,
            model=model,
            prompt=prompt,
            voice_map=chunk_voices,
            alias_map=chunk_aliases,
        )
        if pcm_parts:
            pcm_parts.append(silence_pcm(0.2))
        pcm_parts.append(pcm)
        if sleep_s > 0 and ci < len(chunks) - 1:
            time.sleep(sleep_s)

    if dry_run:
        return out_path

    pcm_to_wav(b"".join(pcm_parts), out_path)
    print(f"  -> {out_path} ({out_path.stat().st_size} bytes)")

    meta = {
        "story_id": story_id,
        "title": story.get("title"),
        "type": story.get("type"),
        "mode": "full",
        "chunks": len(chunks),
        "model": model,
        "tone": tone,
        "context": context,
        "voices": voice_map,
        "personas": persona_map,
        "aliases": full_alias_map,
        "file": out_path.name,
        "bytes": out_path.stat().st_size,
    }
    meta_path = out_dir / f"{story_id}.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {meta_path.name}")
    return out_path


def generate_per_line(
    *,
    story: dict[str, Any],
    out_dir: Path,
    tone: str,
    context: str,
    voice_map: dict[str, str],
    persona_map: dict[str, str],
    model: str,
    fmt: str,
    sleep_s: float,
    start: int,
    end: int | None,
    dry_run: bool,
    skip_existing: bool,
    upload: bool,
    bucket: str,
    client: Any,
    types_mod: Any,
) -> None:
    lines = story.get("lines") or []
    if not lines:
        raise SystemExit(f"{story['id']} has no lines.")

    start_i = max(0, start)
    end_i = len(lines) if end is None else min(len(lines), end)
    if start_i >= end_i:
        raise SystemExit(f"Empty line range: start={start_i} end={end_i}")

    story_dir = out_dir / story["id"]
    story_dir.mkdir(parents=True, exist_ok=True)

    print(f"Story: {story['id']} - {story.get('title')}")
    print(f"Type:  {story.get('type')} | lines {start_i}..{end_i - 1} of {len(lines)}")
    print("Voices / personas:")
    for sp, voice in voice_map.items():
        print(f"  {sp} -> {voice}")
        print(f"      {persona_map.get(sp, '')}")
    if tone:
        print(f"Tone: {tone}")
    if context:
        print(f"Context: {context}")
    print(f"Output: {story_dir}")

    manifest_lines: list[dict[str, Any]] = []
    content_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    for i in range(start_i, end_i):
        line = lines[i]
        text = (line.get("kr") or "").strip()
        if not text:
            print(f"  [{i:02d}] skip - empty kr")
            continue

        speaker = line.get("speaker") or "Narrator"
        voice = voice_map[speaker]
        persona = persona_map[speaker]
        out_name = f"line_{i:02d}.{fmt}"
        out_path = story_dir / out_name
        prompt = build_prompt(
            text=text,
            speaker=speaker,
            voice=voice,
            story=story,
            tone=tone,
            context=context,
            persona=persona,
            line_index=i,
            total_lines=len(lines),
        )

        entry = {
            "index": i,
            "file": out_name,
            "storage_path": f"{story['id']}/{out_name}",
            "speaker": speaker,
            "voice": voice,
            "persona": persona,
            "kr": text,
            "en": line.get("en"),
        }

        if skip_existing and out_path.exists():
            print(f"  [{i:02d}] exists — {out_name}")
            entry["status"] = "skipped_existing"
            manifest_lines.append(entry)
            continue

        if dry_run:
            print(f"\n--- line_{i:02d} ({speaker} / {voice}) ---")
            print(prompt)
            entry["status"] = "dry_run"
            manifest_lines.append(entry)
            continue

        print(f"  [{i:02d}] synthesizing ({speaker} / {voice})...", flush=True)
        try:
            pcm = synthesize_line(
                client,
                types_mod,
                model=model,
                prompt=prompt,
                voice=voice,
            )
            wav_path = story_dir / f"line_{i:02d}.wav"
            pcm_to_wav(pcm, wav_path)
            if fmt == "mp3":
                wav_to_mp3(wav_path, out_path)
                wav_path.unlink(missing_ok=True)
            entry["status"] = "ok"
            entry["bytes"] = out_path.stat().st_size
            print(f"       -> {out_path.name} ({entry['bytes']} bytes)")
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"       ERROR: {exc}", file=sys.stderr)

        manifest_lines.append(entry)
        if sleep_s > 0 and i < end_i - 1:
            time.sleep(sleep_s)

    manifest = {
        "story_id": story["id"],
        "title": story.get("title"),
        "type": story.get("type"),
        "model": model,
        "format": fmt,
        "tone": tone,
        "context": context,
        "voices": voice_map,
        "personas": persona_map,
        "lines": manifest_lines,
    }
    manifest_path = story_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {manifest_path}")

    if upload and not dry_run:
        print(f"Uploading to Supabase bucket '{bucket}'...")
        remote = upload_folder(
            story_dir,
            bucket=bucket,
            story_id=story["id"],
            content_type=content_type,
        )
        for path in remote:
            print(f"  uploaded {path}")

    ok = sum(1 for e in manifest_lines if e.get("status") == "ok")
    err = sum(1 for e in manifest_lines if e.get("status") == "error")
    print(f"\nDone. ok={ok} errors={err} total_attempted={len(manifest_lines)}")
    if err:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini 2.5 Flash TTS audio for one or more stories."
    )
    parser.add_argument(
        "story_ids",
        nargs="+",
        help="Story id(s), e.g. ST_001 or ST_007 ST_008 ST_009 ST_010",
    )
    parser.add_argument(
        "--stories",
        type=Path,
        default=DEFAULT_STORIES,
        help=f"Path to stories JSON (default: {DEFAULT_STORIES})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"Output root (default: {DEFAULT_FULL_OUT} with --full, "
            f"else {DEFAULT_OUT})"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Generate one full-story WAV per id as {out}/{ST_00X}.wav "
            "(multi-speaker when 2 cast members)"
        ),
    )
    parser.add_argument(
        "--tone",
        default=None,
        help="Delivery tone, e.g. 'polite workplace; manager firm, employee nervous'",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Extra scene context for the TTS style prompt",
    )
    parser.add_argument(
        "--voice",
        default="Kore",
        help="Default / narrator voice (default: Kore)",
    )
    parser.add_argument(
        "--voice-map",
        default=None,
        help="Speaker→voice map, e.g. '팀장님=Kore,지민=Puck'",
    )
    parser.add_argument(
        "--persona-map",
        default=None,
        help=(
            "Speaker identity locks separated by '|', e.g. "
            "'팀장님=Korean male manager, mid-40s, lower pitch, composed|"
            "지민=Korean female employee, mid-20s, lighter pitch, apologetic'"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"TTS model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--format",
        choices=("mp3", "wav"),
        default="mp3",
        help="Per-line output format (ignored with --full; default: mp3)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Pause between line/story requests in seconds (default: 0.4)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Per-line mode: start line index inclusive (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Per-line mode: end line index exclusive (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stories/lines that already have an output file",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated per-line files to Supabase Storage after generation",
    )
    parser.add_argument(
        "--bucket",
        default="story-audio",
        help="Supabase storage bucket (default: story-audio)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and voice map without calling the API",
    )
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        parser.error("Set GEMINI_API_KEY in the environment or .env")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_FULL_OUT if args.full else DEFAULT_OUT

    interactive = sys.stdin.isatty() and len(args.story_ids) == 1 and not args.full
    tone = prompt_if_missing(args.tone, "Tone", interactive=interactive)
    context = prompt_if_missing(args.context, "Context", interactive=interactive)
    voice_overrides = parse_voice_map(args.voice_map)
    persona_overrides = parse_kv_map(args.persona_map, "--persona-map", sep="|")

    client = None
    types_mod = None
    if not args.dry_run:
        genai, types_mod = get_genai()
        client = genai.Client(api_key=api_key)

    for idx, story_id in enumerate(args.story_ids):
        if idx > 0 and args.sleep > 0 and not args.dry_run:
            time.sleep(args.sleep)

        story = load_story(args.stories, story_id)
        voice_map = assign_voices(story, voice_overrides, args.voice)
        persona_map = assign_personas(story, persona_overrides)

        if args.full:
            try:
                generate_full_story(
                    story=story,
                    out_dir=out_dir,
                    tone=tone,
                    context=context,
                    voice_map=voice_map,
                    persona_map=persona_map,
                    model=args.model,
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                    client=client,
                    types_mod=types_mod,
                    sleep_s=args.sleep,
                )
            except Exception as exc:
                print(f"ERROR {story_id}: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            if len(args.story_ids) > 1:
                parser.error("Per-line mode accepts one story_id (or pass --full).")
            generate_per_line(
                story=story,
                out_dir=out_dir,
                tone=tone,
                context=context,
                voice_map=voice_map,
                persona_map=persona_map,
                model=args.model,
                fmt=args.format,
                sleep_s=args.sleep,
                start=args.start,
                end=args.end,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                upload=args.upload,
                bucket=args.bucket,
                client=client,
                types_mod=types_mod,
            )

    if args.full and not args.dry_run:
        print("\nDone. Full WAVs ready — split with:")
        print(
            "  python split_story_audio.py ST_00X "
            "--out-dir story_audio --model small --format mp3 "
            "--end-boost 0.1 --boost-from 0"
        )


if __name__ == "__main__":
    main()
