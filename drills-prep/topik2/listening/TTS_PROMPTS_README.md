# TOPIK II Listening TTS

## API scripts (use these)

```bash
# Pilot (7 files) — Gemini
./kr-dict-env/bin/python generate_listening_tts.py --provider gemini

# Pilot (7 files) — OpenAI
./kr-dict-env/bin/python generate_listening_tts.py --provider openai

# One id
./kr-dict-env/bin/python generate_listening_tts.py --provider gemini --id ai2l_visual_match_001

# Full unique pool later
./kr-dict-env/bin/python generate_listening_tts.py --provider gemini --all --mp3
```

Env in `.env`:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

Outputs:
- `drills_audio/topik2/listening/gemini/{id}.wav`
- `drills_audio/topik2/listening/openai/{id}.wav`

Voice locks:
| Speaker | Gemini | OpenAI |
|---|---|---|
| Man (남자) | Charon | onyx |
| Woman (여자) | Kore | nova |

Gemini uses native multi-speaker. OpenAI stitches per-turn WAVs (no native 2-speaker).

## Prompt docs (reference / manual UI fallback)

| File | Use |
|---|---|
| `TTS_MASTER_PROMPT_GEMINI.md` | Gemini role prompt |
| `TTS_MASTER_PROMPT_CHATGPT.md` | ChatGPT/OpenAI role prompt |
| `tts_pilot.json` | Pilot batch list |
| `pilot_prompts/` | Optional UI paste prompts |
