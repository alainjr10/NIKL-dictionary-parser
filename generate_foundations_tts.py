#!/usr/bin/env python3
"""Generate Hanuri Foundations Hangul TTS audio (OpenAI or Gemini).

Reads texts from foundations_tts_items.json and writes:
  foundations_audio/audio/FD_XXX.mp3
  foundations_audio/manifest.json
  foundations_audio/README.md

Examples:
  # OpenAI — fill only missing files (default provider)
  ./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --pending

  # One clip (skip if exists)
  ./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_073

  # Regenerate specific clips
  ./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_017 --id FD_021 --force

  # Dry-run
  ./.venv/Scripts/python.exe generate_foundations_tts.py --dry-run --pending

  # Gemini (when quota recovers)
  ./.venv/Scripts/python.exe generate_foundations_tts.py --provider gemini --pending

Env:
  OPENAI_API_KEY   (openai)
  GEMINI_API_KEY   (gemini)
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
import unicodedata
import wave
from datetime import date
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
ITEMS_JSON = ROOT / "foundations_tts_items.json"
DEFAULT_OUT = ROOT / "foundations_audio"

GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_VOICE = "Kore"

OPENAI_MODEL = "gpt-4o-mini-tts"
OPENAI_VOICE = "nova"  # clear adult female; good tutor fit

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
TRAIL_SILENCE_SEC = 0.20
MIN_FILE_BYTES = 1024

DEFAULT_SLEEP = {"openai": 0.4, "gemini": 2.0}
RETRY_SLEEP = 3.0
RATE_LIMIT_SLEEP = {"openai": 15.0, "gemini": 30.0}
MAX_ATTEMPTS = 6

ID_RE = re.compile(r"^FD_(\d{3})$", re.IGNORECASE)

ROLE_PROMPT = """You are a warm Korean tutor recording premium study audio for absolute beginners
learning Hangul (Hanuri Foundations: Hangul Mastery + Day-One Korean).

ROLE
- Speak ONLY the Korean text given below. Audio only — no English, no intro, no outro.
- Do NOT say “자”, “들어보세요”, item numbers, or any extra words.
- Voice: clear Seoul-standard (표준어) adult woman, warm tutor quality, mid pitch.
- Pace: slightly slow and crisp (~85–90% of normal conversation), especially for single syllables.
- One short utterance only.

PRONUNCIATION CARE
- Minimal pairs must stay distinct: 아 vs 어, 오 vs 우, 가 vs 카, 애 vs 에, 왜 vs 웨, 예 vs 얘.
- For 의: careful citation-form syllable [의], NOT reduced particle [에].
- For 외 / 왜 / 웨: keep each as distinct as possible.
- For single syllables (가, 어, 왜): say once, cleanly, with natural ending (no rush).
- For phrases (안녕하세요): natural polite intonation, not robotic.
"""

OPENAI_INSTRUCTIONS_SYLLABLE = (
    "Speak clear Seoul-standard Korean as a warm adult tutor. "
    "Say this Hangul syllable or short word ONCE only, slowly and crisply "
    "(~85–90% of normal pace). No English, no intro/outro, no extra words. "
    "Keep minimal pairs distinct (아/어, 애/에, 왜/웨, 의 as citation [의] not [에])."
)

OPENAI_INSTRUCTIONS_PHRASE = (
    "Speak clear Seoul-standard Korean as a warm adult tutor. "
    "Natural polite intonation, slightly slow and crisp. "
    "Say the phrase exactly once. No English, no intro/outro, no extra words."
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


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def silence_pcm(seconds: float, rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


def wav_to_mp3(wav_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
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


def write_wav(path: Path, pcm: bytes, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or SAMPLE_RATE
        return frames / float(rate)


def mp3_duration_sec(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_pcm_gemini(response: Any) -> bytes:
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


def unit_hints_for(text: str, kind: str) -> list[str]:
    if kind == "phrase":
        return ["day_one", "quiz"]

    words = {
        "가방",
        "사람",
        "사랑",
        "안녕",
        "친구",
        "커피",
        "학교",
        "한국",
    }
    if text in words:
        return ["words", "quiz"]

    vowels = {
        "아",
        "애",
        "야",
        "얘",
        "어",
        "에",
        "여",
        "예",
        "오",
        "와",
        "왜",
        "외",
        "요",
        "우",
        "워",
        "웨",
        "위",
        "유",
        "으",
        "의",
        "이",
    }
    if text in vowels:
        if text in {"와", "왜", "외", "워", "웨", "위", "의"}:
            return ["vowels_complex", "quiz"]
        return ["vowels_basic", "quiz"]

    if text in {"각", "간", "갈", "감", "갑", "강", "물", "밥"}:
        return ["consonants_batchim", "quiz"]

    if text in {"카", "타", "파", "차"}:
        return ["consonants_aspirated", "quiz"]

    return ["consonants_plain", "quiz"]


def build_items(texts: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, raw in enumerate(texts, start=1):
        text = nfc(raw.strip())
        item_id = f"FD_{i:03d}"
        kind = "phrase" if i >= 52 else "syllable"
        items.append(
            {
                "id": item_id,
                "text": text,
                "kind": kind,
                "file": f"audio/{item_id}.mp3",
                "unit_hints": unit_hints_for(text, kind),
            }
        )
    return items


def normalize_ids(raw_ids: list[str] | None) -> list[str] | None:
    if not raw_ids:
        return None
    out: list[str] = []
    for raw in raw_ids:
        token = raw.strip().upper()
        m = ID_RE.match(token)
        if not m:
            raise SystemExit(f"Invalid id {raw!r}; expected FD_001 … FD_074")
        n = int(m.group(1))
        if n < 1 or n > 74:
            raise SystemExit(f"Id out of range: {token}")
        out.append(f"FD_{n:03d}")
    # preserve order, unique
    seen: set[str] = set()
    uniq: list[str] = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def synthesize_gemini(
    client: Any,
    types_mod: Any,
    *,
    model: str,
    voice: str,
    text: str,
    kind: str,
) -> tuple[bytes, int]:
    pace_note = (
        "Single Hangul syllable / short word. Say it ONCE, slowly and crisply "
        "(~85–90% pace). Tiny natural trailing pause is fine."
        if kind == "syllable"
        else "Short Day-One Korean phrase. Natural polite intonation, slightly slow and clear."
    )
    prompt = "\n".join(
        [
            ROLE_PROMPT.strip(),
            "",
            pace_note,
            "Do not answer in text — audio only.",
            "",
            "TEXT",
            text,
        ]
    )
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
    return extract_pcm_gemini(response), SAMPLE_RATE


def synthesize_openai(
    client: Any,
    *,
    model: str,
    voice: str,
    text: str,
    kind: str,
) -> tuple[bytes, int]:
    instructions = (
        OPENAI_INSTRUCTIONS_SYLLABLE
        if kind == "syllable"
        else OPENAI_INSTRUCTIONS_PHRASE
    )
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
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
                f"Unexpected WAV from OpenAI: ch={wf.getnchannels()} "
                f"width={wf.getsampwidth()} rate={rate}"
            )
        frames = wf.readframes(wf.getnframes())
    return frames, rate


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
        )
    )


def validate_audio(
    mp3_path: Path,
    *,
    kind: str,
    wav_path: Path | None = None,
) -> None:
    size = mp3_path.stat().st_size
    if size < MIN_FILE_BYTES:
        raise RuntimeError(f"Audio too small ({size} bytes)")

    duration: float | None = None
    if wav_path and wav_path.exists():
        duration = wav_duration_sec(wav_path)
    if duration is None:
        duration = mp3_duration_sec(mp3_path)
    if duration is None:
        return

    lo = 0.25
    hi = 6.0 if kind == "phrase" else 4.0
    if duration < lo or duration > hi:
        raise RuntimeError(
            f"Duration out of range: {duration:.2f}s (expected {lo}-{hi}s for {kind})"
        )


def write_manifest(
    out_dir: Path,
    items: list[dict[str, Any]],
    *,
    voice: str,
    model: str,
    provider: str,
) -> Path:
    payload = {
        "version": 1,
        "bucket_hint": "foundations-audio",
        "provider": provider,
        "voice": voice,
        "model": model,
        "items": items,
    }
    path = out_dir / "manifest.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_readme(
    out_dir: Path,
    *,
    provider: str,
    voice: str,
    model: str,
    count: int,
    generated: int,
    skipped: int,
    failed: list[str],
) -> Path:
    lines = [
        "# Hanuri Foundations audio",
        "",
        "Premium beginner Hangul / Day-One Korean TTS clips for the Hanuri Flutter app.",
        "",
        f"- **Provider:** `{provider}`",
        f"- **Voice:** `{voice}`",
        f"- **Model:** `{model}`",
        f"- **Date:** {date.today().isoformat()}",
        f"- **Items:** {count}",
        f"- **Last run:** generated={generated}, skipped={skipped}, failed={len(failed)}",
        "",
        "## Layout",
        "",
        "```",
        "foundations_audio/",
        "  audio/FD_001.mp3 … FD_074.mp3",
        "  manifest.json",
        "  README.md",
        "```",
        "",
        "Lookup key = exact Korean `text` (NFC). Filenames are ASCII IDs only.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "# Missing only (OpenAI)",
        "./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --pending",
        "",
        "# One id",
        "./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_073",
        "",
        "# Force overwrite specific ids",
        "./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_017 --id FD_021 --force",
        "```",
        "",
        "## Notes",
        "",
        "- Bare jamo (ㄱ, ㅏ, …) are **not** spoken alone; syllable examples are used instead.",
        "- Syllable clips include ~200ms trailing silence.",
        "- Re-run is idempotent: existing MP3s are skipped unless `--force` (or `--id` + `--force`).",
        "",
    ]
    if failed:
        lines.extend(
            [
                "## Failed IDs (last run)",
                "",
                ", ".join(failed),
                "",
            ]
        )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_texts() -> list[str]:
    data = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    texts = [nfc(t) for t in data["items"]]
    if len(texts) != 74:
        raise SystemExit(f"Expected 74 items, got {len(texts)} from {ITEMS_JSON}")
    if len(set(texts)) != len(texts):
        raise SystemExit("Duplicate texts in foundations_tts_items.json")
    return texts


def select_planned(
    items: list[dict[str, Any]],
    *,
    out_dir: Path,
    ids: list[str] | None,
    pending: bool,
    force: bool,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    by_id = {it["id"]: it for it in items}

    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"Unknown id(s): {', '.join(missing)}")
        planned = [by_id[i] for i in ids]
    else:
        planned = items[offset:]
        if limit is not None:
            planned = planned[:limit]

    if pending and not force:
        planned = [
            it
            for it in planned
            if not (out_dir / it["file"]).exists()
        ]
    return planned


def make_client(provider: str) -> tuple[Any, Any | None]:
    """Return (client, types_mod_or_None)."""
    load_env()
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY in .env or the environment.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise SystemExit("Install google-genai: pip install google-genai") from exc
        return genai.Client(api_key=api_key), types

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install openai: pip install openai") from exc
    return OpenAI(api_key=api_key), None


def generate_all(
    *,
    provider: str = "openai",
    out_dir: Path = DEFAULT_OUT,
    model: str | None = None,
    voice: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    pending: bool = False,
    ids: list[str] | None = None,
    sleep_sec: float | None = None,
    limit: int | None = None,
    offset: int = 0,
    keep_wav: bool = False,
) -> dict[str, Any]:
    if provider == "gemini":
        model = model or GEMINI_MODEL
        voice = voice or GEMINI_VOICE
    else:
        model = model or OPENAI_MODEL
        voice = voice or OPENAI_VOICE

    if sleep_sec is None:
        sleep_sec = DEFAULT_SLEEP[provider]

    texts = load_texts()
    items = build_items(texts)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    planned = select_planned(
        items,
        out_dir=out_dir,
        ids=ids,
        pending=pending,
        force=force,
        offset=offset,
        limit=limit,
    )

    # Always refresh full manifest (all 74 rows).
    write_manifest(
        out_dir, items, voice=voice, model=model, provider=provider
    )

    if dry_run:
        print(f"Dry-run [{provider}]: {len(planned)} item(s) → {out_dir}")
        for it in planned:
            exists = (out_dir / it["file"]).exists()
            flag = "EXISTS" if exists else "NEW"
            print(
                f"  {it['id']}  {it['text']!r}  → {it['file']}  "
                f"({it['kind']}, {flag})"
            )
        write_readme(
            out_dir,
            provider=provider,
            voice=voice,
            model=model,
            count=len(items),
            generated=0,
            skipped=0,
            failed=[],
        )
        return {
            "generated": 0,
            "skipped": 0,
            "failed": [],
            "planned": len(planned),
            "out_dir": str(out_dir),
        }

    if not planned:
        print("Nothing to generate (all selected outputs already exist).")
        print("Tip: pass --force to overwrite, or --id FD_XXX --force for one clip.")
        print(f"Out: {out_dir}")
        write_readme(
            out_dir,
            provider=provider,
            voice=voice,
            model=model,
            count=len(items),
            generated=0,
            skipped=0,
            failed=[],
        )
        return {
            "generated": 0,
            "skipped": 0,
            "failed": [],
            "out_dir": str(out_dir),
        }

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH (required for MP3 output).")

    client, types_mod = make_client(provider)

    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print(f"Voice:    {voice}")
    print(f"Items:    {len(planned)} (of {len(items)} total)")
    print(f"Force:    {force}")
    print(f"Out:      {out_dir}")
    print(f"Sleep:    {sleep_sec}s between requests")
    print()

    generated = 0
    skipped = 0
    failed: list[str] = []
    rate_base = RATE_LIMIT_SLEEP[provider]

    for i, it in enumerate(planned, 1):
        item_id = it["id"]
        text = it["text"]
        kind = it["kind"]
        mp3_path = out_dir / it["file"]
        wav_path = mp3_path.with_suffix(".wav")

        if mp3_path.exists() and not force:
            print(f"[{i}/{len(planned)}] skip {item_id} ({text}) — exists")
            skipped += 1
            continue

        print(f"[{i}/{len(planned)}] {item_id} {text!r} ...", flush=True)

        last_err: BaseException | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                if provider == "gemini":
                    pcm, rate = synthesize_gemini(
                        client,
                        types_mod,
                        model=model,
                        voice=voice,
                        text=text,
                        kind=kind,
                    )
                else:
                    pcm, rate = synthesize_openai(
                        client,
                        model=model,
                        voice=voice,
                        text=text,
                        kind=kind,
                    )

                if not pcm or len(pcm) < 500:
                    raise RuntimeError("Empty/corrupt PCM from TTS")

                if kind == "syllable":
                    pcm = pcm + silence_pcm(TRAIL_SILENCE_SEC, rate=rate)

                write_wav(wav_path, pcm, rate=rate)
                wav_to_mp3(wav_path)
                validate_audio(mp3_path, kind=kind, wav_path=wav_path)

                if not keep_wav:
                    wav_path.unlink(missing_ok=True)

                size = mp3_path.stat().st_size
                print(f"  wrote {mp3_path.name} ({size} bytes)")
                generated += 1
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wav_path.unlink(missing_ok=True)
                if mp3_path.exists():
                    try:
                        mp3_path.unlink()
                    except OSError:
                        pass

                if attempt >= MAX_ATTEMPTS - 1:
                    print(f"  FAILED: {exc}")
                    break

                if is_rate_limit_error(exc):
                    wait = rate_base * (2**attempt)
                    print(
                        f"  rate-limited (attempt {attempt + 1}/{MAX_ATTEMPTS}); "
                        f"sleeping {wait:.0f}s then retry..."
                    )
                    time.sleep(wait)
                    continue

                print(f"  retry after error: {exc}")
                time.sleep(RETRY_SLEEP)

        if last_err is not None:
            failed.append(item_id)

        if sleep_sec > 0 and i < len(planned):
            time.sleep(sleep_sec)

    write_manifest(
        out_dir, items, voice=voice, model=model, provider=provider
    )
    write_readme(
        out_dir,
        provider=provider,
        voice=voice,
        model=model,
        count=len(items),
        generated=generated,
        skipped=skipped,
        failed=failed,
    )

    print()
    print(f"Done. generated={generated} skipped={skipped} failed={len(failed)}")
    if failed:
        print("Failed ids:", ", ".join(failed))

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "out_dir": str(out_dir),
        "manifest": str(out_dir / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hanuri Foundations Hangul TTS (OpenAI or Gemini)."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        default="openai",
        help="TTS provider (default: openai)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output root (default: foundations_audio/)",
    )
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument("--voice", default=None, help="Override voice id")
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        metavar="FD_XXX",
        help="Only this id (repeatable). Use with --force to regenerate.",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Only generate ids whose MP3 is missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing MP3s",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Pause between requests (seconds). Defaults: openai=0.4, gemini=2.0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items this run (ignored when --id is set)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many items from the start (ignored when --id is set)",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep WAV intermediates after MP3 conversion",
    )
    args = parser.parse_args()

    if args.offset < 0:
        sys.exit("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        sys.exit("--limit must be >= 1")
    if args.sleep is not None and args.sleep < 0:
        sys.exit("--sleep must be >= 0")

    ids = normalize_ids(args.ids)
    # Convenience: --id alone with an existing file should not silently no-op
    # unless user forgot --force — select_planned keeps it, then loop skips.
    # If --pending and no --id, only missing. If neither, all (then skip exists).

    result = generate_all(
        provider=args.provider,
        out_dir=args.out_dir,
        model=args.model,
        voice=args.voice,
        dry_run=args.dry_run,
        force=args.force,
        pending=args.pending,
        ids=ids,
        sleep_sec=args.sleep,
        limit=args.limit,
        offset=args.offset,
        keep_wav=args.keep_wav,
    )
    if result.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
