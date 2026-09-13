# Hanuri Foundations audio

Premium beginner Hangul / Day-One Korean TTS clips for the Hanuri Flutter app.

- **Provider:** `openai`
- **Voice:** `nova`
- **Model:** `gpt-4o-mini-tts`
- **Date:** 2026-09-13
- **Items:** 74
- **Last run:** generated=5, skipped=0, failed=0

## Layout

```
foundations_audio/
  audio/FD_001.mp3 … FD_074.mp3
  manifest.json
  README.md
```

Lookup key = exact Korean `text` (NFC). Filenames are ASCII IDs only.

## Regenerate

```bash
# Missing only (OpenAI)
./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --pending

# One id
./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_073

# Force overwrite specific ids
./.venv/Scripts/python.exe generate_foundations_tts.py --provider openai --id FD_017 --id FD_021 --force
```

## Notes

- Bare jamo (ㄱ, ㅏ, …) are **not** spoken alone; syllable examples are used instead.
- Syllable clips include ~200ms trailing silence.
- Re-run is idempotent: existing MP3s are skipped unless `--force` (or `--id` + `--force`).
