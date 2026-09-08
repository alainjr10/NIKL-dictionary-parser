#!/usr/bin/env python3
"""Split a full-story TTS audio file into per-line clips using forced text alignment.

Uses faster-whisper word timestamps + ordered matching against the known Korean
lines in stories_content.json (no silence detection).

Usage:
  python split_story_audio.py ST_007
  python split_story_audio.py ST_001 --audio story_audio/ST_001.wav
  python split_story_audio.py ST_001 --audio story_audio/ST_001-full.wav

Outputs:
  story_audio/ST_001/line_00.mp3
  ...
  story_audio/ST_001/split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_STORIES = Path(__file__).parent / "res" / "stories_content.json"
DEFAULT_OUT = Path(__file__).parent / "story_audio"
DEFAULT_AUDIO_DIR = Path(__file__).parent / "story_audio"


def resolve_audio_path(story_id: str, audio: Path | None, audio_dir: Path) -> Path:
    """Prefer ST_00X.wav, then legacy ST_00X-full.wav."""
    if audio is not None:
        return audio
    candidates = [
        audio_dir / f"{story_id}.wav",
        audio_dir / f"{story_id}-full.wav",
        audio_dir / f"{story_id}.mp3",
        audio_dir / f"{story_id}-full.mp3",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise SystemExit(
        f"No audio found for {story_id}. Pass --audio or place one of:\n  {tried}"
    )


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class LineSpan:
    index: int
    speaker: str
    kr: str
    start: float
    end: float
    score: float
    matched_asr: str


def load_story(stories_path: Path, story_id: str) -> dict[str, Any]:
    data = json.loads(stories_path.read_text(encoding="utf-8"))
    for story in data:
        if story.get("id") == story_id:
            return story
    raise SystemExit(f"Story {story_id!r} not found in {stories_path}")


def normalize_ko(text: str) -> str:
    """Strip spaces/punct so ASR text can match script text."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\uac00-\ud7a3]+", "", text, flags=re.UNICODE)
    return text


def transcribe_words(audio_path: Path, model_size: str, device: str) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "Missing dependency. Install with:\n"
            "  pip install faster-whisper\n"
        )

    print(f"Loading Whisper model '{model_size}' on {device}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language="ko",
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=True,
    )
    print(f"Detected language={info.language} probability={info.language_probability:.2f}")

    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            # fallback: whole segment as one "word"
            if seg.text and seg.text.strip():
                words.append(Word(seg.text.strip(), float(seg.start), float(seg.end)))
            continue
        for w in seg.words:
            token = (w.word or "").strip()
            if not token:
                continue
            words.append(Word(token, float(w.start), float(w.end)))

    if not words:
        raise SystemExit("No words transcribed from audio. Check the file / model.")
    print(f"Transcribed {len(words)} words, "
          f"span {words[0].start:.2f}s -> {words[-1].end:.2f}s")
    return words


def align_lines(words: list[Word], lines: list[dict[str, Any]]) -> list[LineSpan]:
    """Greedily assign consecutive ASR words to each known story line in order."""
    norms = [normalize_ko(w.text) for w in words]
    # Drop empty normalized tokens but keep index map
    keep = [(i, n) for i, n in enumerate(norms) if n]
    if not keep:
        raise SystemExit("ASR produced no usable Korean tokens.")

    cursor = 0  # index into keep[]
    spans: list[LineSpan] = []

    for li, line in enumerate(lines):
        target = normalize_ko(line.get("kr") or "")
        if not target:
            raise SystemExit(f"Line {li} has empty kr text")

        remaining_lines = len(lines) - li
        remaining_keep = len(keep) - cursor
        if remaining_keep <= 0:
            raise SystemExit(f"Ran out of ASR words before line {li}")

        # Leave at least 1 token per remaining line after this one
        max_end = len(keep) - (remaining_lines - 1)
        best_end = cursor + 1
        best_score = -1.0

        for end in range(cursor + 1, max_end + 1):
            chunk = "".join(n for _, n in keep[cursor:end])
            if not chunk:
                continue
            ratio = SequenceMatcher(None, chunk, target).ratio()
            # Prefer length-similar matches
            length_ratio = min(len(chunk), len(target)) / max(len(chunk), len(target))
            score = 0.75 * ratio + 0.25 * length_ratio

            # Soft bonus when chunk covers most of target
            if len(chunk) >= len(target) * 0.85:
                score += 0.02

            if score > best_score:
                best_score = score
                best_end = end

            # Early stop: chunk already much longer than target and score falling
            if len(chunk) > len(target) * 1.5 and score < best_score - 0.05:
                break

        word_idxs = [keep[i][0] for i in range(cursor, best_end)]
        start_t = words[word_idxs[0]].start
        end_t = words[word_idxs[-1]].end
        matched = "".join(words[i].text for i in word_idxs)

        spans.append(
            LineSpan(
                index=li,
                speaker=line.get("speaker") or "Narrator",
                kr=line.get("kr") or "",
                start=start_t,
                end=end_t,
                score=best_score,
                matched_asr=matched,
            )
        )
        cursor = best_end

    # Stretch last line to audio end if a little tail remains
    if spans and words:
        spans[-1].end = max(spans[-1].end, words[-1].end)

    # Fix tiny overlaps / ensure monotonic
    for i in range(1, len(spans)):
        if spans[i].start < spans[i - 1].end:
            mid = (spans[i].start + spans[i - 1].end) / 2
            spans[i - 1].end = mid
            spans[i].start = mid

    return spans


def probe_duration(audio_path: Path) -> float:
    """Return true media duration in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def apply_cut_windows(
    spans: list[LineSpan],
    *,
    pad_start: float,
    pad_end: float,
    audio_end: float,
    end_boost: float = 0.0,
    boost_from: int = 0,
    max_overlap: float = 0.08,
) -> list[tuple[float, float]]:
    """Build cut windows.

    Whisper often ends late turns early. ``end_boost`` pushes those endings
    later; the next clip starts where the previous ended (no double-count /
    no leftover at the next clip's head).
    """
    windows: list[tuple[float, float]] = []
    prev_end: float | None = None

    for i, sp in enumerate(spans):
        if prev_end is None:
            start = max(0.0, sp.start - pad_start)
        else:
            # Handoff at previous cut end so boosted tails aren't repeated
            start = prev_end

        boost = end_boost if i >= boost_from else 0.0
        desired_end = sp.end + pad_end + boost

        if i + 1 < len(spans):
            next_sp = spans[i + 1]
            gap = next_sp.start - sp.end
            if boost <= 0 and gap > 0.05:
                end = min(desired_end, next_sp.start - 0.02)
            elif boost <= 0:
                end = min(desired_end, next_sp.start + max_overlap)
            else:
                # Boosted: take the extra time; leave the next line enough room
                latest = next_sp.end - 0.35
                end = min(desired_end, latest, audio_end)
                end = max(end, sp.end + 0.05)
        else:
            end = audio_end  # always keep true file tail on last line

        end = min(audio_end, end)
        if end <= start:
            end = start + 0.05
        windows.append((start, end))
        prev_end = end

    return windows


def ffmpeg_cut(
    audio_path: Path,
    out_path: Path,
    start: float,
    end: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Input-seek first then accurate ss/to for cleaner edges
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-acodec",
        "libmp3lame" if out_path.suffix.lower() == ".mp3" else "pcm_s16le",
    ]
    if out_path.suffix.lower() == ".mp3":
        cmd += ["-q:a", "2"]
    cmd.append(str(out_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split full story TTS audio into per-line clips via text alignment."
    )
    parser.add_argument("story_id", help="Story id, e.g. ST_001")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Full dialogue audio (default: story_audio/ST_00X.wav)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory used to auto-find ST_00X.wav (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--stories",
        type=Path,
        default=DEFAULT_STORIES,
        help=f"Stories JSON (default: {DEFAULT_STORIES})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="faster-whisper model size: tiny/base/small/medium/large-v3 (default: small)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--format",
        choices=("mp3", "wav"),
        default="mp3",
        help="Output clip format (default: mp3)",
    )
    parser.add_argument(
        "--pad-start",
        type=float,
        default=0.05,
        help="Seconds to include before each line start (default: 0.05)",
    )
    parser.add_argument(
        "--pad-end",
        type=float,
        default=0.12,
        help="Base seconds after each aligned end (default: 0.12)",
    )
    parser.add_argument(
        "--end-boost",
        type=float,
        default=0.1,
        help="Extra seconds added to line endings from --boost-from (default: 0.1)",
    )
    parser.add_argument(
        "--boost-from",
        type=int,
        default=0,
        help="Line index from which --end-boost applies (default: 0 = all lines)",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.08,
        help="Max overlap into next turn when end-boost is 0 (default: 0.08)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=None,
        help="If set, use this value for both --pad-start and --pad-end",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Warn if alignment score for a line is below this (default: 0.45)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Align and print timestamps only; do not cut files",
    )
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.story_id, args.audio, args.audio_dir)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    story = load_story(args.stories, args.story_id)
    lines = story.get("lines") or []
    if not lines:
        raise SystemExit(f"{args.story_id} has no lines")

    story_dir = args.out_dir / args.story_id
    story_dir.mkdir(parents=True, exist_ok=True)

    print(f"Story: {args.story_id} - {story.get('title')}")
    print(f"Audio: {audio_path}")
    print(f"Lines: {len(lines)}")

    words = transcribe_words(audio_path, args.model, args.device)
    spans = align_lines(words, lines)

    print("\nAligned spans:")
    low = 0
    for sp in spans:
        flag = ""
        if sp.score < args.min_score:
            flag = "  << LOW SCORE"
            low += 1
        print(
            f"  [{sp.index:02d}] {sp.start:6.2f}-{sp.end:6.2f}s  "
            f"score={sp.score:.2f}  {sp.speaker}{flag}"
        )
        print(f"       script: {sp.kr}")
        print(f"       asr:    {sp.matched_asr}")

    pad_start = args.pad if args.pad is not None else args.pad_start
    pad_end = args.pad if args.pad is not None else args.pad_end
    probed = probe_duration(audio_path)
    audio_end = probed if probed > 0 else (words[-1].end if words else spans[-1].end)
    if probed > 0:
        print(f"Media duration (ffprobe): {probed:.3f}s")

    windows = apply_cut_windows(
        spans,
        pad_start=pad_start,
        pad_end=pad_end,
        audio_end=audio_end,
        end_boost=args.end_boost,
        boost_from=args.boost_from,
        max_overlap=args.max_overlap,
    )

    print(
        f"\nCut windows (pad_start={pad_start:.2f}s, pad_end={pad_end:.2f}s, "
        f"end_boost={args.end_boost:.2f}s from line {args.boost_from:02d}):"
    )
    for sp, (w_start, w_end) in zip(spans, windows):
        print(f"  [{sp.index:02d}] cut {w_start:6.2f}-{w_end:6.2f}s")

    if args.dry_run:
        print("\nDry-run only; no files written.")
        return

    cut_rows: list[dict[str, Any]] = []
    for sp, (w_start, w_end) in zip(spans, windows):
        out_name = f"line_{sp.index:02d}.{args.format}"
        out_path = story_dir / out_name
        print(f"Cutting {out_name}...")
        ffmpeg_cut(audio_path, out_path, w_start, w_end)
        cut_rows.append(
            {
                "index": sp.index,
                "file": out_name,
                "speaker": sp.speaker,
                "kr": sp.kr,
                "align_start": round(sp.start, 3),
                "align_end": round(sp.end, 3),
                "cut_start": round(w_start, 3),
                "cut_end": round(w_end, 3),
                "score": round(sp.score, 3),
                "matched_asr": sp.matched_asr,
                "bytes": out_path.stat().st_size,
            }
        )

    manifest = {
        "story_id": args.story_id,
        "source_audio": str(audio_path),
        "model": args.model,
        "device": args.device,
        "format": args.format,
        "pad_start": pad_start,
        "pad_end": pad_end,
        "end_boost": args.end_boost,
        "boost_from": args.boost_from,
        "max_overlap": args.max_overlap,
        "audio_end": round(audio_end, 3),
        "low_score_lines": low,
        "lines": cut_rows,
    }
    manifest_path = story_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {manifest_path}")
    if low:
        print(
            f"Warning: {low} line(s) had low alignment scores. "
            "Listen and adjust --model (try medium) or re-export cleaner audio."
        )
    print("Done.")


if __name__ == "__main__":
    main()
