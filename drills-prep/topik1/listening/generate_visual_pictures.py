#!/usr/bin/env python3
"""Generate TOPIK I listening picture questions (Q15–16) as 2x2 exam sheets.

One image call per item, then crop into img1..img4 and stamp ①–④.
Output: drills_images/topik1/listening/pic_q15_001/{panel_raw,panel,img1..img4}.png

Usage:
  ./.venv/Scripts/python.exe drills-prep/topik1/listening/generate_visual_pictures.py --id pic_q15_001
  ./.venv/Scripts/python.exe drills-prep/topik1/listening/generate_visual_pictures.py --all --provider gemini
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
POOL = Path(__file__).resolve().parent / "pool.json"
OUT_ROOT = ROOT / "drills_images" / "topik1" / "listening"

GEMINI_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-exp-image-generation",
)
OPENAI_MODELS = (
    "gpt-image-1",
    "gpt-image-1-mini",
    "gpt-image-1.5",
    "gpt-image-2",
    "dall-e-3",
)

STYLE = """
You are drawing official TOPIK I (한국어능력시험) Listening exam pictures.
Match the look of a printed TOPIK workbook page: simple grayscale cartoon scenes.

STYLE (strict):
- Black-and-white / grayscale line illustration (exam workbook look)
- Clean outlines PLUS light-to-moderate gray shading on clothes, hair, ground,
  and soft shadows under people/objects
- Plain white / off-white background
- ABSOLUTELY NO text or glyphs anywhere: no Korean, no English, no letters,
  no digits, no circled numbers, no speech bubbles, no watermarks, no logos,
  no arrows, no brand names. Any signs, menus, screens, or maps must be blank.
- Korean everyday setting. Clear, readable actions. Calm expressions.
- Same line weight and character design in all 4 panels

LAYOUT (strict):
- Exactly ONE image containing a 2x2 grid of four separate scenes
- Top-left = panel 1, top-right = panel 2, bottom-left = panel 3, bottom-right = panel 4
- Equal panel sizes, thin black borders between panels, white margins
- Leave the top-left corner of each panel empty (option numbers are added later)
- Each panel shows ONE clear action
"""


def picture_items() -> list[dict]:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    return [q for q in pool if q.get("visual_kind") == "picture"]


def load_item(item_id: str) -> dict:
    wanted = item_id if item_id.startswith("ai1l_") else f"ai1l_{item_id}"
    item = next((q for q in picture_items() if q["id"] == wanted), None)
    if item is None:
        raise SystemExit(f"Unknown picture id: {item_id}")
    if not item.get("image_specs"):
        raise SystemExit(f"{wanted} has no image_specs")
    return item


def slug_dir(item: dict) -> str:
    # pool images are "pic_q15_001/img1"
    return item["images"][0].split("/")[0]


def build_prompt(item: dict) -> str:
    lines = [
        STYLE.strip(),
        "",
        "Draw these four panels. Foils are near-misses: related place or people,",
        "but the KEY ACTION must match only that panel's description.",
        "Do not swap actions between panels. No writing. No option numbers.",
        "",
        "Panel actions (follow EXACTLY):",
    ]
    for i, spec in enumerate(item["image_specs"], start=1):
        kind = "target action" if spec.upper().startswith("CORRECT") else "different action"
        scene = re.sub(r"^(FOIL|CORRECT):\s*", "", spec, flags=re.I).strip()
        lines.append(f"Panel {i} ({kind}): {scene}")
    lines.append("")
    lines.append(
        "The target-action panel must be unmistakable from the other three. "
        "Do not write labels such as correct, foil, or target. "
        "Blank signs, screens, and papers only."
    )
    return "\n".join(lines)


def generate_sheet_gemini(prompt: str) -> Image.Image:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last_err: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            print(f"  gemini model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
            for cand in response.candidates or []:
                for part in cand.content.parts or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        data = inline.data
                        raw = data if isinstance(data, (bytes, bytearray)) else bytes(data)
                        return Image.open(BytesIO(raw)).convert("RGB")
            raise RuntimeError("No image in Gemini response")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  fail {model}: {exc}")
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def generate_sheet_openai(prompt: str) -> Image.Image:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    last_err: Exception | None = None
    for model in OPENAI_MODELS:
        try:
            print(f"  openai model: {model}")
            kwargs: dict = {
                "model": model,
                "prompt": prompt,
                "size": "1024x1024",
            }
            if model.startswith("gpt-image"):
                kwargs["quality"] = "medium"
            elif model == "dall-e-3":
                kwargs["quality"] = "standard"
                kwargs["n"] = 1
            result = client.images.generate(**kwargs)
            item = result.data[0]
            if getattr(item, "b64_json", None):
                raw = base64.b64decode(item.b64_json)
                return Image.open(BytesIO(raw)).convert("RGB")
            if getattr(item, "url", None):
                import urllib.request

                with urllib.request.urlopen(item.url) as resp:
                    return Image.open(BytesIO(resp.read())).convert("RGB")
            raise RuntimeError("No image payload in OpenAI response")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  fail {model}: {exc}")
    raise RuntimeError(f"All OpenAI image models failed: {last_err}")


def crop_quadrants(sheet: Image.Image) -> list[Image.Image]:
    w, h = sheet.size
    mx, my = int(w * 0.03), int(h * 0.03)
    sheet = sheet.crop((mx, my, w - mx, h - my))
    w, h = sheet.size
    mid_x, mid_y = w // 2, h // 2
    gap = max(2, int(min(w, h) * 0.01))
    boxes = [
        (0, 0, mid_x - gap, mid_y - gap),
        (mid_x + gap, 0, w, mid_y - gap),
        (0, mid_y + gap, mid_x - gap, h),
        (mid_x + gap, mid_y + gap, w, h),
    ]
    return [sheet.crop(b) for b in boxes]


def _load_font(size: int):
    for path in (
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def stamp_option_number(im: Image.Image, index: int) -> Image.Image:
    label = ["①", "②", "③", "④"][index]
    font_size = max(26, im.size[0] // 14)
    font = _load_font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(6, im.size[0] // 50)
    label_gap = max(4, pad // 2)
    margin_top = th + label_gap + pad
    margin_left = max(tw + label_gap + pad // 2, margin_top // 2)
    out = Image.new(
        "RGB",
        (im.size[0] + margin_left + pad, im.size[1] + margin_top + pad),
        "white",
    )
    ox, oy = margin_left, margin_top
    out.paste(im.convert("RGB"), (ox, oy))
    draw = ImageDraw.Draw(out)
    draw.rectangle(
        [ox, oy, ox + im.size[0] - 1, oy + im.size[1] - 1],
        outline="black",
        width=max(1, im.size[0] // 200),
    )
    tx = max(pad // 2, ox - tw - label_gap)
    ty = max(pad // 2, oy - th - label_gap)
    draw.text((tx, ty), label, font=font, fill="black")
    return out


def compose_numbered_panel(quads: list[Image.Image]) -> Image.Image:
    w = max(q.size[0] for q in quads)
    h = max(q.size[1] for q in quads)
    quads = [q.resize((w, h), Image.Resampling.LANCZOS) for q in quads]
    gap = max(16, w // 36)
    sheet = Image.new("RGB", (w * 2 + gap * 3, h * 2 + gap * 3), "white")
    positions = [
        (gap, gap),
        (gap * 2 + w, gap),
        (gap, gap * 2 + h),
        (gap * 2 + w, gap * 2 + h),
    ]
    for q, pos in zip(quads, positions):
        sheet.paste(q, pos)
    return sheet


def _show(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def save_sheet(sheet: Image.Image, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_panel = out_dir / "panel_raw.png"
    sheet.save(raw_panel, "PNG")
    print(f"wrote {_show(raw_panel)}  ({sheet.size[0]}x{sheet.size[1]})")
    quads = crop_quadrants(sheet)
    numbered = [stamp_option_number(im, i) for i, im in enumerate(quads)]
    for i, im in enumerate(numbered, start=1):
        p = out_dir / f"img{i}.png"
        im.save(p, "PNG")
        print(f"wrote {_show(p)}")
    panel = compose_numbered_panel(numbered)
    panel_path = out_dir / "panel.png"
    panel.save(panel_path, "PNG")
    print(f"wrote {_show(panel_path)}  ({panel.size[0]}x{panel.size[1]})")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="e.g. pic_q15_001 or ai1l_pic_q15_001")
    ap.add_argument("--all", action="store_true", help="All 12 picture items")
    ap.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    ap.add_argument("--force", action="store_true", help="Regenerate even if panel.png exists")
    ap.add_argument("--sleep", type=float, default=2.0, help="Pause between items")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override output root (default: drills_images/topik1/listening)",
    )
    args = ap.parse_args()
    if not args.all and not args.id:
        raise SystemExit("Pass --id or --all")

    items = picture_items() if args.all else [load_item(args.id)]
    failed: list[str] = []
    for i, item in enumerate(items, start=1):
        base = Path(args.out_root) if args.out_root else OUT_ROOT
        if not base.is_absolute():
            base = (ROOT / base).resolve()
        out_dir = base / slug_dir(item)
        if (out_dir / "panel.png").exists() and not args.force:
            print(f"[{i}/{len(items)}] skip {item['id']} (exists)")
            continue
        print(f"[{i}/{len(items)}] {item['id']} answer={item['answer'] + 1}")
        try:
            prompt = build_prompt(item)
            if args.provider == "openai":
                sheet = generate_sheet_openai(prompt)
            else:
                sheet = generate_sheet_gemini(prompt)
            save_sheet(sheet, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            failed.append(item["id"])
        if args.sleep and i < len(items):
            time.sleep(args.sleep)
    print(f"done. failed={len(failed)}")
    if failed:
        print("failed ids:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
