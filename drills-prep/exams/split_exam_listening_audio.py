#!/usr/bin/env python3
"""Split a full TOPIK listening exam MP3 into per-question (or per-passage) clips.

Uses faster-whisper word timestamps + ordered matching against spoken
transcripts in listening.json (same idea as split_story_audio.py).

Shared consecutive transcripts (e.g. Q25–Q26) collapse to one passage clip;
both questions get the same audio path.

Usage:
  ./.venv/Scripts/python.exe drills-prep/exams/split_exam_listening_audio.py topik1_91
  ./.venv/Scripts/python.exe drills-prep/exams/split_exam_listening_audio.py topik1_91 --model small
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXAMS = Path(__file__).resolve().parent


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class PassageSpan:
    key: str
    question_ids: list[int]
    spoken: str
    start: float
    end: float
    score: float
    matched_asr: str


def normalize_ko(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\uac00-\ud7a3]+", "", text, flags=re.UNICODE)
    return text


def spoken_from_transcript(transcript: str) -> str:
    """Keep only lines that are actually spoken on the track."""
    parts: list[str] = []
    for raw in (transcript or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Drop blank-response placeholders: "여자: ( )"
        body = re.sub(r"^(남자|여자|남|여)\s*:\s*", "", line).strip()
        if not body or body in {"( )", "()", "(…)"}:
            continue
        # Strip stage directions / SFX cues in parentheses
        body = re.sub(r"\([^)]*\)", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            parts.append(body)
    return " ".join(parts)


def load_exam(exam_id: str) -> tuple[Path, Path, list[dict[str, Any]]]:
    folder = EXAMS / exam_id
    listening = folder / "listening.json"
    if not listening.exists():
        raise SystemExit(f"Missing {listening}")
    audio_candidates = [
        folder / "audio-full.mp3",
        folder / "audioi-full.mp3",  # typo-tolerant
        folder / "audio_full.mp3",
        folder / f"{exam_id}_listening_full.mp3",
    ]
    audio = next((p for p in audio_candidates if p.exists()), None)
    if audio is None:
        raise SystemExit(
            "No full audio found. Place one of:\n  "
            + "\n  ".join(str(p) for p in audio_candidates)
        )
    data = json.loads(listening.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("listening.json must be a non-empty list")
    return folder, audio, data


def build_passages(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive items that share the same spoken transcript."""
    passages: list[dict[str, Any]] = []
    for q in questions:
        spoken = spoken_from_transcript(q.get("transcript") or "")
        if not spoken:
            raise SystemExit(f"Q{q.get('id')} has no spoken transcript text")
        qid = int(q["id"])
        if passages and passages[-1]["spoken"] == spoken:
            passages[-1]["question_ids"].append(qid)
        else:
            passages.append(
                {
                    "key": f"q{qid:02d}",
                    "question_ids": [qid],
                    "spoken": spoken,
                }
            )
    # Rename keys for multi-id passages
    for p in passages:
        ids = p["question_ids"]
        if len(ids) == 1:
            p["key"] = f"q{ids[0]:02d}"
        else:
            p["key"] = f"q{ids[0]:02d}-{ids[-1]:02d}"
    return passages


def transcribe_words(audio_path: Path, model_size: str, device: str) -> list[Word]:
    from faster_whisper import WhisperModel

    print(f"Loading Whisper '{model_size}' on {device}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language="ko",
        word_timestamps=True,
        vad_filter=False,
        # Long exam tracks hallucinate/collapse mid-file when conditioning
        # on prior text (seen on topik1_96 after the intro music bed).
        condition_on_previous_text=False,
    )
    print(
        f"Detected language={info.language} "
        f"probability={info.language_probability:.2f}"
    )
    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            if seg.text and seg.text.strip():
                words.append(Word(seg.text.strip(), float(seg.start), float(seg.end)))
            continue
        for w in seg.words:
            token = (w.word or "").strip()
            if token:
                words.append(Word(token, float(w.start), float(w.end)))
    if not words:
        raise SystemExit("No words transcribed from audio.")
    print(
        f"Transcribed {len(words)} words, "
        f"span {words[0].start:.2f}s -> {words[-1].end:.2f}s"
    )
    return words


NUM_ANNOUNCE_RE = re.compile(r"^(\d+)\s*번\.?$")
# Whisper sometimes drops 번 → bare "13." / "2."
BARE_NUM_RE = re.compile(r"^(\d+)\.?$")
# Hangul Sino-Korean: 이십오번 / ASR typo 이심육번
_HANGUL_DIGIT = {
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_HANGUL_BUN_RE = re.compile(
    r"^(삼십|이십|십)?(일|이|삼|사|오|육|칠|팔|구)?번\.?$"
)


def parse_question_number_token(token: str) -> int | None:
    """Parse '25번', '25.', or Hangul '이십오번' / '이심육번'."""
    token = (token or "").strip().replace("심", "십")
    m = NUM_ANNOUNCE_RE.match(token)
    if m:
        return int(m.group(1))
    m2 = BARE_NUM_RE.match(token)
    if m2 and "." in token:
        return int(m2.group(1))
    m3 = _HANGUL_BUN_RE.match(token)
    if not m3:
        return None
    tens_s, ones_s = m3.group(1), m3.group(2)
    if not tens_s and not ones_s:
        return None
    n = 0
    if tens_s == "삼십":
        n = 30
    elif tens_s == "이십":
        n = 20
    elif tens_s == "십":
        n = 10
    if ones_s:
        n += _HANGUL_DIGIT[ones_s]
    return n if n >= 1 else None


def _collect_num_candidates(
    words: list[Word], max_q: int
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Return (N번 times, bare 'N.' times) separately — bare is fallback only."""
    bun: dict[int, list[float]] = {i: [] for i in range(1, max_q + 1)}
    bare: dict[int, list[float]] = {i: [] for i in range(1, max_q + 1)}
    for i, w in enumerate(words):
        token = (w.text or "").strip()
        # Digit N번
        m = NUM_ANNOUNCE_RE.match(token)
        if m:
            n = int(m.group(1))
            if 1 <= n <= max_q:
                bun[n].append(float(w.start))
            continue
        # Hangul N번 (이십오번 / 이심육번)
        n_h = parse_question_number_token(token)
        if n_h is not None and 1 <= n_h <= max_q and "번" in token and not token[0].isdigit():
            bun[n_h].append(float(w.start))
            continue
        if i + 1 < len(words):
            a = token
            b = (words[i + 1].text or "").strip()
            ma = BARE_NUM_RE.match(a)
            if ma and b.startswith("번") and not b.startswith("번입"):
                n = int(ma.group(1))
                if 1 <= n <= max_q and words[i + 1].start - w.start < 1.0:
                    bun[n].append(float(w.start))
                    continue
        m2 = BARE_NUM_RE.match(token)
        if m2 and "." in token:
            n = int(m2.group(1))
            if 1 <= n <= max_q:
                bare[n].append(float(w.start))
    return bun, bare


def find_question_number_anchors(
    words: list[Word],
    max_q: int,
    passages: list[dict[str, Any]] | None = None,
) -> dict[int, float]:
    """Find 'N번' announcements in order.

    Tolerates Whisper dropping/mishearing some numbers by (1) accepting bare
    ``N.`` tokens, (2) spoken-transcript fallback, (3) linear interpolation.
    """
    bun, bare = _collect_num_candidates(words, max_q)
    anchors: dict[int, float] = {}
    prev_t = 0.0
    min_gap = 8.0

    def _pick(cands: list[float], after: float) -> float | None:
        gap = min_gap if after > 0 else 0.0
        for t in cands:
            if t >= after + gap:
                return t
        return None

    for n in range(1, max_q + 1):
        picked = _pick(bun[n], prev_t)
        src = "asr"
        if picked is None:
            picked = _pick(bare[n], prev_t)
            src = "bare-num"
        if picked is not None:
            anchors[n] = picked
            prev_t = picked
            print(f"  anchor Q{n} @ {picked:.1f}s ({src})")
        else:
            print(f"  anchor Q{n} MISSING after {prev_t:.1f}s")

    # Spoken / structural fallback for gaps
    spoken_by_q: dict[int, str] = {}
    pair_mates: dict[int, list[int]] = {}  # qid -> full pair id list
    if passages:
        for p in passages:
            ids = [int(x) for x in p["question_ids"]]
            for qid in ids:
                spoken_by_q[qid] = p.get("spoken") or ""
                if len(ids) >= 2:
                    pair_mates[qid] = ids

    replays = find_replay_cues(words)
    pair_cues = find_pair_section_cues(words)

    def _near_replay(t: float, tol: float = 4.0) -> bool:
        return any(abs(t - r) <= tol for r in replays)

    still_missing = [n for n in range(1, max_q + 1) if n not in anchors]
    for n in still_missing:
        prev_known = max((k for k in anchors if k < n), default=None)
        lo = (anchors[prev_known] + 1.0) if prev_known is not None else 1.0
        hi_candidates = [anchors[k] for k in range(n + 1, max_q + 1) if k in anchors]

        mates = pair_mates.get(n)
        is_pair = mates is not None
        is_pair_leader = is_pair and n == mates[0]

        if is_pair_leader:
            # Pair section cue is the *start* of the shared passage, not the hi bound.
            # Answer N번 sits after both plays, before the *next* section cue / anchor.
            cues_after = [c for c in pair_cues if c > lo]
            # Ignore intro-style cues that land before this exam's late pairs
            # (keep cues after the previous question's anchor).
            section_start = cues_after[0] if cues_after else lo
            later_cues = [c for c in pair_cues if c > section_start + 10.0]
            hi_opts = later_cues + hi_candidates
            hi = (
                min(hi_opts) - 1.0
                if hi_opts
                else (words[-1].end if words else section_start + 180.0)
            )
            last_speech = section_start
            for w in words:
                if section_start < w.start < hi and "다시" not in w.text:
                    last_speech = max(last_speech, w.end)
            guess = min(hi - 8.0, max(section_start + 40.0, last_speech + 8.0))
            anchors[n] = guess
            print(
                f"  anchor Q{n} @ {anchors[n]:.1f}s "
                f"(pair-answer after section@{section_start:.1f}, before {hi + 1:.1f})"
            )
            continue

        cue_hi = [c for c in pair_cues if c > lo]
        # For solos, don't treat pair-section cues as hard ends unless they're
        # the only bound (avoids collapsing into early intro cues).
        hi_opts = hi_candidates if hi_candidates else cue_hi
        hi = (
            min(hi_opts) - 1.0
            if hi_opts
            else (words[-1].end if words else lo + 120)
        )

        if is_pair and n != mates[0]:
            # Second of pair: a bit after the leader (filled later if leader pending)
            continue

        # Don't search inside the previous question's second play
        if prev_known is not None:
            prev_a = anchors[prev_known]
            prev_replays = [r for r in replays if prev_a < r < hi]
            if prev_replays:
                pr = prev_replays[0]
                first_len = pr - prev_a
                # Only treat as this question's 다시 if first play looks typical
                if 8.0 <= first_len <= 55.0:
                    lo = max(lo, pr + first_len + 2.0)

        spoken = spoken_by_q.get(n, "")
        hit = find_spoken_start(words, spoken, lo, hi) if spoken else None
        if hit is not None and not _near_replay(hit):
            anchors[n] = max(lo, hit - 2.5)
            print(f"  anchor Q{n} @ {anchors[n]:.1f}s (spoken@{hit:.1f})")

    # Pair trailers (26/28/30) once leaders exist
    for n in range(1, max_q + 1):
        if n in anchors:
            continue
        mates = pair_mates.get(n)
        if not mates or n == mates[0]:
            continue
        leader = mates[0]
        if leader not in anchors:
            continue
        nxt = min(
            ([anchors[k] for k in range(n + 1, max_q + 1) if k in anchors] or [anchors[leader] + 40])
        )
        anchors[n] = min(anchors[leader] + 20.0, nxt - 2.0)
        print(f"  anchor Q{n} @ {anchors[n]:.1f}s (pair-trailer)")

    # Interpolate any remaining holes between known neighbors
    still_missing = [n for n in range(1, max_q + 1) if n not in anchors]
    for n in still_missing:
        left = max((k for k in anchors if k < n), default=None)
        right = min((k for k in anchors if k > n), default=None)
        if left is None or right is None:
            continue
        # Place proportionally by question index between left/right
        span_q = right - left
        span_t = anchors[right] - anchors[left]
        anchors[n] = anchors[left] + span_t * ((n - left) / span_q)
        print(f"  anchor Q{n} @ {anchors[n]:.1f}s (interpolated)")

    if len(anchors) < max_q:
        missing = [i for i in range(1, max_q + 1) if i not in anchors]
        raise SystemExit(
            f"Only found {len(anchors)}/{max_q} question anchors. Missing: {missing[:12]}"
        )

    # Ensure strictly increasing (spoken/interp can collide)
    for n in range(2, max_q + 1):
        if anchors[n] <= anchors[n - 1] + 1.0:
            anchors[n] = anchors[n - 1] + 5.0
            print(f"  anchor Q{n} nudged to {anchors[n]:.1f}s (was <= Q{n - 1})")

    return anchors


def find_replay_cues(words: list[Word]) -> list[float]:
    """Times where the announcer says 다시 들으십시오."""
    cues: list[float] = []
    for i, w in enumerate(words[:-1]):
        if "다시" in w.text and "들으" in words[i + 1].text:
            cues.append(float(w.start))
    return cues


def find_pair_section_cues(words: list[Word]) -> list[float]:
    """'다음을 듣고 물음에 답하십시오' (shared-passage section)."""
    cues: list[float] = []
    for i, w in enumerate(words):
        if not w.text.startswith("다음을"):
            continue
        window = "".join(x.text for x in words[i : i + 8])
        if "듣고" in window and ("물음" in window or "답하" in window):
            cues.append(float(w.start))
    return cues


def find_spoken_start(
    words: list[Word],
    spoken: str,
    search_after: float,
    search_before: float,
) -> float | None:
    """Earliest good transcript match in (search_after, search_before)."""
    target = normalize_ko(spoken)
    if not target:
        return None
    if search_before <= search_after + 1.0:
        return None

    # Short cues: prefer an exact first-token hit (avoids fuzzy false positives)
    raw_parts = [p for p in re.split(r"\s+", spoken.strip()) if p]
    if raw_parts and len(target) <= 16:
        needle = normalize_ko(raw_parts[0])
        if len(needle) >= 2:
            for w in words:
                if not (search_after < w.start < search_before):
                    continue
                if needle in normalize_ko(w.text):
                    return float(w.start)

    # Prefer a distinctive head chunk (first ~20 chars) for locating play 1
    head = target[: max(12, min(24, len(target)))]

    norms = [(w, normalize_ko(w.text)) for w in words]
    keep = [(w, n) for w, n in norms if n and search_after < w.start < search_before]
    if not keep:
        return None

    best: tuple[float, float] | None = None  # (score, start)
    min_ratio = 0.72 if len(target) <= 20 else 0.75
    for s in range(len(keep)):
        chunk = ""
        for e in range(s, min(len(keep), s + 40)):
            chunk += keep[e][1]
            if len(chunk) < len(head) * 0.6:
                continue
            ratio = SequenceMatcher(None, chunk[: len(head) + 8], head).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, keep[s][0].start)
            if len(chunk) > len(head) * 2:
                break
        if best and best[0] >= min_ratio:
            return best[1]
    if best and best[0] >= 0.62:
        return best[1]
    return None


def build_spans_from_anchors(
    passages: list[dict[str, Any]],
    anchors: dict[int, float],
    words: list[Word],
    audio_end: float,
) -> tuple[float, list[PassageSpan]]:
    """Build cut spans.

    Solo items (TOPIK I Q1–24 style): ``N번`` → next ``N번`` (covers both plays).

    Paired items (Q25+): shared passage is played *before* the answer ``N번``
    announcements. Start at section cue / first transcript hit; end at the next
    pair's passage start (not at ``25번``, which is mid-block after both plays).
    """
    intro_end = max(0.0, anchors[1] - 0.05)
    replays = find_replay_cues(words)
    pair_cues = find_pair_section_cues(words)
    print(f"  replay cues: {[round(x, 1) for x in replays]}")
    print(f"  pair section cues: {[round(x, 1) for x in pair_cues]}")

    # First pass: compute each passage start
    starts: list[float] = []
    for i, passage in enumerate(passages):
        ids = passage["question_ids"]
        is_pair = len(ids) >= 2
        prev_end = starts[i - 1] if i else intro_end

        if not is_pair:
            starts.append(anchors[ids[0]])
            continue

        # Pair: passage is played twice *before* answer numbers (25번/26번).
        # Structure: [section cue?] [play1] [다시 들으십시오] [play2] [N번] [N+1번]
        answer_t = anchors[ids[0]]
        if i and len(passages[i - 1]["question_ids"]) == 1:
            # Previous solo also has play×2; don't search from its N번 start
            # or pair start collapses onto the solo.
            prev_id = passages[i - 1]["question_ids"][0]
            prev_a = anchors[prev_id]
            prev_replays = [r for r in replays if prev_a < r < answer_t]
            if prev_replays:
                pr = prev_replays[0]
                first_len = max(15.0, pr - prev_a)
                search_lo = pr + first_len + 3.0
            else:
                search_lo = prev_a + 40.0
        else:
            search_lo = (starts[i - 1] + 0.5) if i else (intro_end + 0.5)
        search_lo = min(search_lo, answer_t - 15.0)

        cue_candidates = [c for c in pair_cues if search_lo <= c < answer_t]
        replays_before = [r for r in replays if search_lo < r < answer_t]
        spoken_t = find_spoken_start(
            words, passage["spoken"], search_lo, answer_t - 1.0
        )

        if not replays_before:
            # No replay found — fall back to cue / spoken / gap
            if cue_candidates:
                start = cue_candidates[0]
            elif spoken_t is not None:
                start = max(search_lo, spoken_t - 1.0)
            else:
                start = max(search_lo, answer_t - 120.0)
            method = "fallback"
        else:
            replay = replays_before[-1]
            second_len = max(10.0, answer_t - replay)
            estimated = max(search_lo, replay - second_len - 0.5)

            # spoken hit only counts if it lands in the first-play half
            spoken_ok = (
                spoken_t is not None and search_lo <= spoken_t <= replay - 5.0
            )

            if cue_candidates:
                start = cue_candidates[0]
                method = "section-cue"
            elif spoken_ok:
                start = max(search_lo, float(spoken_t) - 1.0)
                method = "spoken"
            else:
                start = estimated
                method = "replay-symmetric"

        print(
            f"  pair {passage['key']} start={start:.1f}s via {method} "
            f"(answer {ids[0]}번@{answer_t:.1f}"
            f"{f', replay@{replays_before[-1]:.1f}' if replays_before else ''}"
            f"{f', spoken@{spoken_t:.1f}' if spoken_t is not None else ''}"
            f"{f', cues={[round(c,1) for c in cue_candidates]}' if cue_candidates else ''})"
        )

        starts.append(start)

    # Second pass: ends = next start (or audio end)
    spans: list[PassageSpan] = []
    for i, passage in enumerate(passages):
        ids = passage["question_ids"]
        start = starts[i]
        if i + 1 < len(passages):
            end = starts[i + 1] - 0.05
        else:
            last_word_end = words[-1].end if words else audio_end
            end = min(audio_end, max(last_word_end + 1.0, start + 5.0))

        # Solo that precedes a pair: end at pair start (already starts[i+1])
        # Solo mid-exam: prefer next question's N번 if earlier than next start
        if len(ids) == 1 and i + 1 < len(passages):
            next_ids = passages[i + 1]["question_ids"]
            if len(next_ids) == 1:
                end = anchors[next_ids[0]] - 0.05

        if end <= start:
            end = start + 1.0

        spans.append(
            PassageSpan(
                key=passage["key"],
                question_ids=list(ids),
                spoken=passage["spoken"],
                start=start,
                end=end,
                score=1.0,
                matched_asr="pair-aware" if len(ids) >= 2 else f"{ids[0]}번->next",
            )
        )
        print(
            f"  span {passage['key']} ids={ids} "
            f"{start:.1f}-{end:.1f}s ({end - start:.1f}s)"
        )

    return intro_end, spans


def save_words_cache(path: Path, words: list[Word]) -> None:
    data = [{"t": w.text, "s": w.start, "e": w.end} for w in words]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_words_cache(path: Path) -> list[Word]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Word(x["t"], float(x["s"]), float(x["e"])) for x in data]

def probe_duration(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def cut_windows(
    spans: list[PassageSpan],
    *,
    audio_end: float,
    pad_start: float,
    pad_end: float,
) -> list[tuple[float, float]]:
    """Apply small pads; spans already run 번→next번 (both plays included)."""
    windows: list[tuple[float, float]] = []
    for i, sp in enumerate(spans):
        start = max(0.0, sp.start - pad_start)
        end = min(audio_end, sp.end + pad_end)
        if i + 1 < len(spans):
            end = min(end, spans[i + 1].start - 0.05)
        end = max(end, start + 0.3)
        windows.append((start, end))
    return windows


def ffmpeg_cut(audio_path: Path, out_path: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        "libmp3lame",
        "-q:a",
        "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])


def main() -> None:
    ap = argparse.ArgumentParser(description="Split TOPIK exam listening full audio")
    ap.add_argument("exam_id", help="e.g. topik1_91")
    ap.add_argument("--model", default="small", help="faster-whisper model size")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--pad-start", type=float, default=0.15)
    ap.add_argument("--pad-end", type=float, default=0.0)
    ap.add_argument(
        "--words-cache",
        type=Path,
        default=None,
        help="Reuse cached Whisper word timestamps JSON (skips re-ASR)",
    )
    ap.add_argument(
        "--update-json",
        action="store_true",
        default=True,
        help="Write audio fields into listening.json (default: true)",
    )
    ap.add_argument("--no-update-json", action="store_false", dest="update_json")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")

    folder, audio_path, questions = load_exam(args.exam_id)
    passages = build_passages(questions)
    print(f"exam={args.exam_id} audio={audio_path.name} questions={len(questions)}")
    print(f"passages={len(passages)} (shared transcripts collapsed)")

    spans = None
    cache_path = args.words_cache or (folder / "audio" / "_whisper_words.json")
    if cache_path.exists():
        print(f"Loading cached words from {cache_path}")
        words = load_words_cache(cache_path)
        print(f"Cached {len(words)} words, span {words[0].start:.2f}s -> {words[-1].end:.2f}s")
    else:
        words = transcribe_words(audio_path, args.model, args.device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_words_cache(cache_path, words)
        print(f"Cached words -> {cache_path}")

    max_q = max(int(q["id"]) for q in questions)
    print("Finding sequential N번 anchors...")
    anchors = find_question_number_anchors(words, max_q, passages=passages)
    duration = probe_duration(audio_path) or (words[-1].end + 1.0)
    intro_end, spans = build_spans_from_anchors(passages, anchors, words, duration)
    windows = cut_windows(
        spans,
        audio_end=duration,
        pad_start=args.pad_start,
        pad_end=args.pad_end,
    )

    out_dir = folder / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intro (exam directions before 1번)
    intro_path = out_dir / "intro.mp3"
    if intro_end > 1.0:
        print(f"cut intro.mp3  0.00-{intro_end:.2f}s  ({intro_end:.1f}s)")
        ffmpeg_cut(audio_path, intro_path, 0.0, intro_end)
    else:
        print("skip intro (too short)")

    # question_id -> relative audio path
    q_audio: dict[int, str] = {}
    manifest_passages: list[dict[str, Any]] = []

    for sp, (start, end) in zip(spans, windows):
        out_name = f"{sp.key}.mp3"
        out_path = out_dir / out_name
        print(f"cut {out_name}  {start:.2f}-{end:.2f}s  ({end-start:.1f}s)")
        ffmpeg_cut(audio_path, out_path, start, end)
        rel = f"audio/{out_name}"
        for qid in sp.question_ids:
            q_audio[qid] = rel
        manifest_passages.append(
            {
                **asdict(sp),
                "cut_start": start,
                "cut_end": end,
                "file": rel,
            }
        )

    manifest = {
        "exam_id": args.exam_id,
        "source_audio": audio_path.name,
        "duration_sec": duration,
        "model": args.model,
        "method": "question_number_anchors",
        "intro": {
            "file": "audio/intro.mp3" if intro_end > 1.0 else None,
            "cut_start": 0.0,
            "cut_end": intro_end,
        },
        "anchors": {str(k): v for k, v in anchors.items()},
        "passages": manifest_passages,
    }
    man_path = out_dir / "split_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {man_path.relative_to(ROOT)}")

    if args.update_json:
        for q in questions:
            qid = int(q["id"])
            if qid in q_audio:
                q["audio"] = q_audio[qid]
        listening_path = folder / "listening.json"
        listening_path.write_text(
            json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"updated {listening_path.relative_to(ROOT)} with audio fields")

    print("done.")


if __name__ == "__main__":
    main()
