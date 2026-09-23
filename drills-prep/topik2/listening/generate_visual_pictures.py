#!/usr/bin/env python3
"""Generate TOPIK visual_match picture questions via Gemini or OpenAI.

Strategy for consistency: one 2x2 exam sheet in a single image call,
then crop into img1..img4 and stamp ①–④.

Usage:
  ./.venv/Scripts/python.exe drills-prep/topik2/listening/generate_visual_pictures.py --id 001 --provider gemini
  ./.venv/Scripts/python.exe drills-prep/topik2/listening/generate_visual_pictures.py --id 002 --provider openai
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
POOL = Path(__file__).resolve().parent / "pool.json"
OUT_ROOT = ROOT / "drills_images" / "topik2" / "listening"

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
You are drawing official TOPIK II (한국어능력시험) Listening exam pictures.

STYLE (strict):
- Black-and-white / grayscale line illustration (exam workbook look)
- Clean outlines PLUS light-to-moderate gray shading on clothes, hair, ground,
  and soft shadows under people/objects (richer than flat line art, still
  printable exam style - not photoreal, not comic ink wash overload)
- Plain white / off-white background
- ABSOLUTELY NO text or glyphs anywhere: no Korean, no English, no letters,
  no digits, no circled numbers, no speech bubbles, no watermarks, no logos,
  no arrows. Any signs or boards must be completely blank rectangles.
- Korean everyday setting, adult characters, calm neutral expressions
- Same line weight and character design language in all 4 panels

LAYOUT (strict):
- Exactly ONE image containing a 2x2 grid of four separate scenes
- Top-left = panel 1, top-right = panel 2, bottom-left = panel 3, bottom-right = panel 4
- Equal panel sizes, thin hairline borders between panels, generous white margins
- Leave panel corners empty (option numbers will be added later in code)
- Each panel shows ONE clear action (near-miss foils, same setting family)
"""


def load_item(num: str) -> dict:
    iid = f"ai2l_visual_match_{int(num):03d}"
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    item = next(x for x in pool if x["id"] == iid)
    if item.get("visual_kind") != "picture":
        raise SystemExit(f"{iid} is not a picture item (got {item.get('visual_kind')})")
    return item


def build_prompt(item: dict) -> str:
    specs = item["image_specs"]
    lines = [
        STYLE.strip(),
        "",
        "Draw these four panels in ONE shared setting family (same place, same characters).",
        "ACTION FIDELITY IS CRITICAL: each panel must depict ONLY the action described.",
        "Do not invent extra helpers, different poses, or swapped actions between panels.",
        "No writing and no option numbers in the image.",
        "Use light-to-moderate gray shading (exam workbook look, not flat stick figures).",
        "",
        "Panel actions (follow EXACTLY):",
    ]
    for i, spec in enumerate(specs, start=1):
        kind = "CORRECT" if spec.upper().startswith("CORRECT") else "FOIL"
        scene = re.sub(r"^(FOIL|CORRECT):\s*", "", spec, flags=re.I).strip()
        scene = scene.replace("lost-and-found", "service")
        scene = scene.replace("Lost and Found", "service")
        scene = scene.replace("\u2014", "-").replace("\u2013", "-")
        # Avoid Korean glyphs in the image prompt (numbers stamped later in code)
        scene = scene.replace("도장", "small personal name stamp/seal")
        scene = scene.replace("도록", "exhibition booklet/catalog")
        scene = scene.replace("휴관", "closed")
        scene = scene.replace("접수", "check-in")
        lines.append(f"Panel {i} [{kind}]: {scene}")
    lines.append("")
    lines.append(
        "Near-miss rule: foils share the same people/place but the KEY ACTION differs. "
        "If a foil says someone is still on the ground, they must remain on the ground. "
        "If a foil says alone / no helper, there must be no helper. "
        "If the correct panel says supporting by the arm while standing up, show that lift/assist clearly. "
        "Blank signboards only - zero glyphs."
    )
    # Extra lock for the known near-miss set (helping elderly after a fall)
    if item["id"].endswith("visual_match_002"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet:",
                "- Panel 1: elderly woman ALONE on a bench; nobody else; she is seated, not fallen.",
                "- Panel 2: elderly woman still sitting/lying on the SIDEWALK/GROUND; man ONLY picks up a dropped HAND BAG or shopping bag from the ground; "
                "he is NOT lifting her; NO first-aid kit, NO medical bag.",
                "- Panel 3: elderly woman still on the GROUND; a woman stands nearby holding ONE phone to her ear (calling); "
                "nobody is lifting the elderly woman; NO first-aid kit.",
                "- Panel 4: young man SUPPORTS the elderly woman BY THE ARM while she STANDS UP from the sidewalk (the helping action).",
                "- Panel 4 must NOT look like casual walking arm-in-arm: she is mid-rise from a fall (one knee/hand still near ground, half-standing), "
                "he is lifting/stabilizing her by the arm.",
                "- Do NOT add medical kits, spilled medicine, or CPR scenes in any panel.",
            ]
        )
    elif item["id"].endswith("visual_match_004"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (Korean restaurant):",
                "- Panel 1 CORRECT: waiter/server placing a steaming stew bowl (김치찌개) on the table for a seated woman; "
                "table service in progress (about to fetch water). NOT paying, NOT kitchen-only, NOT leaving.",
                "- Panel 2: customer at a cashier counter paying / handing money or card - NO table service.",
                "- Panel 3: chef alone in a kitchen stirring a pot - NO customer at a dining table.",
                "- Panel 4: customer walking away / exiting with a clear PAPER TAKEOUT bag (food takeaway bag with handles), "
                "NOT a purse, NOT a tote, NOT a backpack.",
                "- Keep the same restaurant family look; blank menus/signs only.",
            ]
        )
    elif item["id"].endswith("visual_match_005"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (hospital / clinic):",
                "- Panel 1: doctor examining a patient's throat in an exam room (tongue depressor / looking in mouth) - NOT reception.",
                "- Panel 2: pharmacist handing a medicine bag over a pharmacy counter - NOT clinic reception.",
                "- Panel 3 CORRECT: male patient standing at a hospital RECEPTION DESK talking to a female receptionist; "
                "waiting-area chairs visible behind him. This is check-in / 접수, NOT examination, NOT pharmacy, NOT hospital bed.",
                "- Panel 4: patient lying in a hospital bed with an IV drip stand - NOT standing at reception.",
                "- Blank signs/charts only; no glyphs.",
            ]
        )
    elif item["id"].endswith("visual_match_006"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (school / campus study area):",
                "- Panel 1: ONE student alone at a podium presenting / speaking to an empty or implied audience - NOT printing.",
                "- Panel 2 CORRECT: TWO students together at a desk with a laptop AND a printer; paper coming out of the printer; "
                "both looking at slides/printouts together. Clear collaboration on printing + checking.",
                "- Panel 3: one student asleep face-down on books in a library - NOT printing, NOT presenting.",
                "- Panel 4: a teacher writing on a blackboard alone (students optional or none) - NOT two students printing.",
                "- Blank screens/boards only; no glyphs.",
            ]
        )
    elif item["id"].endswith("visual_match_008"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (office building):",
                "- Panel 1: employee eating lunch ALONE in a cafeteria - NO documents, NOT walking to a meeting.",
                "- Panel 2: ONE person alone at a photocopy/copy machine - NOT entering a meeting room.",
                "- Panel 3: two colleagues chatting at desks EMPTY-HANDED (no folders/documents) - NOT going to a meeting.",
                "- Panel 4 CORRECT: two colleagues EACH carrying document folders / papers, walking into or through a meeting-room doorway. "
                "Clear intent: heading to a meeting with materials. Blank door plate only (no floor numbers drawn as glyphs if possible).",
                "- Panel 4 critical: BOTH people hold folders; they are mid-step ENTERING an open meeting-room door (door visible). "
                "Not just standing and chatting.",
            ]
        )
    elif item["id"].endswith("visual_match_009"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (bank):",
                "- Panel 1: person alone at an ATM withdrawing cash - NO teller counter interaction.",
                "- Panel 2 CORRECT: female customer at bank teller counter handing an ID card AND a small personal stamp/seal "
                "to a male teller; this is account opening paperwork handoff.",
                "- Panel 3: customer alone at a side desk filling a form - NO teller receiving ID/stamp.",
                "- Panel 4: customer sitting in lobby reading a brochure - NOT at the teller counter.",
                "- Blank posters only; no glyphs.",
            ]
        )
    elif item["id"].endswith("visual_match_010"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (hotel lobby / front desk):",
                "- Panel 1 FOIL ONLY: CLERK hands a keycard TO the GUEST. Guest's hands receive the key. "
                "NO suitcase in panel 1. Direction: clerk -> guest.",
                "- Panel 2: guest sitting in lobby WITH luggage beside them, NOT handing luggage over.",
                "- Panel 3 CORRECT ONLY: Suitcase is ON the front-desk counter (being left for storage). "
                "Clerk holds a small claim TAG out toward the guest. Guest's hand reaches for the TAG. "
                "Do NOT show keycard check-in. The story is luggage storage.",
                "- Panel 4: UNIFORMED BELLHOP (not the guest) pushes a luggage cart INTO an open elevator. Guest is NOT in this panel.",
                "- Blank signs only; no glyphs.",
            ]
        )
    elif item["id"].endswith("visual_match_011"):
        lines.extend(
            [
                "",
                "Mandatory contrasts for this sheet (museum):",
                "- Panel 1 CORRECT ONLY: Museum STAFF (behind desk) extends a thick BOOKLET/CATALOG toward the VISITOR. "
                "Visitor's hands RECEIVE the booklet. Direction must be staff -> visitor, never visitor submitting a form. "
                "Elevator silhouette nearby. Blank wall boards only - zero letters.",
                "- Panel 2: artist painting on an easel outdoors - NOT museum info desk.",
                "- Panel 3: closed museum door with a hanging blank closed-notice placard (NO written word CLOSED, no letters).",
                "- Panel 4: outdoor ticket booth only, no catalog handoff / no indoor info desk. "
                "Ticket booth header board must be a BLANK rectangle - do not write TICKET or any word.",
                "- ZERO letters/digits anywhere in any panel except we leave corners empty for option stamps.",
            ]
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
    raise RuntimeError(
        f"All OpenAI image models failed (project may lack image access): {last_err}"
    )


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
    """Place ①–④ just outside a thin frame around the scene (not inside the art)."""
    label = ["①", "②", "③", "④"][index]
    font_size = max(26, im.size[0] // 14)
    font = _load_font(font_size)

    # Probe label size
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad = max(6, im.size[0] // 50)
    label_gap = max(4, pad // 2)
    # Room above + left so the circled number sits clearly outside the frame
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
    # Frame around the scene only
    draw.rectangle(
        [ox, oy, ox + im.size[0] - 1, oy + im.size[1] - 1],
        outline="black",
        width=max(1, im.size[0] // 200),
    )
    # Number just outside the top-left corner of the frame
    tx = ox - tw - label_gap
    if tx < pad // 2:
        tx = pad // 2
    ty = oy - th - label_gap
    if ty < pad // 2:
        ty = pad // 2
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


def restamp_from_raw(out_dir: Path) -> None:
    """Rebuild img1–4 + panel.png from panel_raw.png with outside numbering."""
    raw_path = out_dir / "panel_raw.png"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    sheet = Image.open(raw_path).convert("RGB")
    quads = crop_quadrants(sheet)
    numbered = [stamp_option_number(im, i) for i, im in enumerate(quads)]
    for i, im in enumerate(numbered, start=1):
        p = out_dir / f"img{i}.png"
        im.save(p, "PNG")
        print(f"wrote {p.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")
    panel = compose_numbered_panel(numbered)
    panel_path = out_dir / "panel.png"
    panel.save(panel_path, "PNG")
    print(f"wrote {panel_path.relative_to(ROOT)}  ({panel.size[0]}x{panel.size[1]})")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="visual_match number, e.g. 002")
    ap.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        default="gemini",
        help="image provider (default: gemini)",
    )
    ap.add_argument(
        "--restamp-only",
        action="store_true",
        help="Rebuild img/panel from panel_raw.png without calling an image API",
    )
    args = ap.parse_args()

    item = load_item(args.id)
    num = f"{int(args.id):03d}"
    out_dir = OUT_ROOT / f"visual_match_{num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.restamp_only:
        print(f"restamp-only: {item['id']}")
        restamp_from_raw(out_dir)
        print("done.")
        return

    prompt = build_prompt(item)
    print(f"item: {item['id']}  answer={item['answer']+1}  provider={args.provider}")
    print("--- prompt ---")
    try:
        print(prompt)
    except UnicodeEncodeError:
        print(prompt.encode("ascii", "replace").decode("ascii"))
    print("--------------")

    if args.provider == "openai":
        sheet = generate_sheet_openai(prompt)
    else:
        sheet = generate_sheet_gemini(prompt)

    raw_panel = out_dir / "panel_raw.png"
    sheet.save(raw_panel, "PNG")
    print(f"wrote {raw_panel.relative_to(ROOT)}  ({sheet.size[0]}x{sheet.size[1]})")

    quads = crop_quadrants(sheet)
    numbered = [stamp_option_number(im, i) for i, im in enumerate(quads)]
    for i, im in enumerate(numbered, start=1):
        p = out_dir / f"img{i}.png"
        im.save(p, "PNG")
        print(f"wrote {p.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")

    panel = compose_numbered_panel(numbered)
    panel_path = out_dir / "panel.png"
    panel.save(panel_path, "PNG")
    print(f"wrote {panel_path.relative_to(ROOT)}  ({panel.size[0]}x{panel.size[1]})")
    print("done.")


if __name__ == "__main__":
    main()
