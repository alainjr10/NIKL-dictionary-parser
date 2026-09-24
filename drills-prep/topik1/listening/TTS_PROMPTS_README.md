# TOPIK I Listening TTS

## Rebuild manifests (after pool edits)

```bash
./.venv/Scripts/python.exe drills-prep/topik1/listening/build_tts_manifest.py
```

## API scripts

```bash
# Pilot (8 files) — Gemini
./.venv/Scripts/python.exe generate_listening_tts.py --level topik1 --provider gemini

# One id
./.venv/Scripts/python.exe generate_listening_tts.py --level topik1 --provider gemini --id ai1l_resp_q1_001

# Incremental unique pool (~153)
./.venv/Scripts/python.exe generate_listening_tts.py --level topik1 --provider gemini --all --limit 5
```

Env in `.env`:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

Outputs:
- `drills_audio/topik1/listening/gemini/{id}.mp3`
- `drills_audio/topik1/listening/openai/{id}.mp3`

Voice locks (same as TOPIK II):
| Speaker | Gemini | OpenAI |
|---|---|---|
| Man (남자) | Charon | onyx |
| Woman (여자) | Kore | nova |

Notes:
- Response / follow-up blanks → spoken prompt only + ~1s silence
- `(딩동댕)` stage cues are stripped (no SFX)
- Paired Q25–30 share one MP3 via `shared_with` / pool `audio` links
