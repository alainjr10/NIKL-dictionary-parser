#!/usr/bin/env python3
"""Generate TOPIK-style single-chart PNGs for visual_match graph items.

Real TOPIK 2×2 sheet pattern:
  ① ②  — same heading, same chart type (near-miss value foils)
  ③ ④  — different heading, different chart type from the top row

Charts always show numeric values (% or counts). Rank in the audio is
reflected by those numbers — never rank-only bars.

Writes:
  drills_images/topik2/listening/visual_match_XXX/imgN.png
  drills_images/topik2/listening/visual_match_XXX/panel.png

Usage:
  ./.venv/Scripts/python.exe drills-prep/topik2/listening/generate_visual_graphs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "drills_images" / "topik2" / "listening"

_FONT_CANDIDATES = (
    "Malgun Gothic",
    "Noto Sans KR",
    "AppleGothic",
    "NanumGothic",
    "Gulim",
)
_PIE_COLORS = ["#4A4A4A", "#5B7C99", "#8FA3B5", "#C5CED6"]


def setup_korean_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise SystemExit(
        "No Korean font found. Install Malgun Gothic / Noto Sans KR, then retry."
    )


def _clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_vbar(
    ax,
    labels: list[str],
    values: list[float],
    title: str,
    *,
    unit: str = "%",
    ymax: float | None = None,
) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    _clean_spines(ax)
    xs = range(len(labels))
    bars = ax.bar(
        xs, values, color="#4A4A4A", width=0.62, edgecolor="black", linewidth=0.6
    )
    ax.set_xticks(list(xs), labels, fontsize=10)
    top = ymax if ymax is not None else (100 if unit == "%" else max(values) * 1.35)
    ax.set_ylim(0, top)
    if unit == "%":
        ax.set_ylabel("%", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for bar, v in zip(bars, values):
        suffix = "%" if unit == "%" else unit
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (1.5 if unit == "%" else top * 0.02),
            f"{v:g}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def draw_hbar(
    ax,
    labels: list[str],
    values: list[float],
    title: str,
    *,
    unit: str = "%",
    xmax: float | None = None,
) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    _clean_spines(ax)
    ys = range(len(labels))
    bars = ax.barh(
        ys, values, color="#5B7C99", height=0.55, edgecolor="black", linewidth=0.6
    )
    ax.set_yticks(list(ys), labels, fontsize=10)
    right = xmax if xmax is not None else (100 if unit == "%" else max(values) * 1.35)
    ax.set_xlim(0, right)
    if unit == "%":
        ax.set_xlabel("%", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        suffix = "%" if unit == "%" else unit
        ax.text(
            bar.get_width() + (1.5 if unit == "%" else right * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{v:g}{suffix}",
            ha="left",
            va="center",
            fontsize=10,
        )


def draw_pie(ax, labels: list[str], values: list[float], title: str) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    colors = _PIE_COLORS[: len(labels)]
    _wedges, _texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        textprops={"fontsize": 10},
        pctdistance=0.65,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_fontweight("bold")
    ax.axis("equal")


def render_option(opt: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = opt["kind"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=160)
    if kind == "vbar":
        draw_vbar(
            ax,
            opt["labels"],
            opt["values"],
            opt["title"],
            unit=opt.get("unit", "%"),
            ymax=opt.get("ymax"),
        )
    elif kind == "hbar":
        draw_hbar(
            ax,
            opt["labels"],
            opt["values"],
            opt["title"],
            unit=opt.get("unit", "%"),
            xmax=opt.get("xmax"),
        )
    elif kind == "pie":
        draw_pie(ax, opt["labels"], opt["values"], opt["title"])
    else:
        raise ValueError(f"unknown kind: {kind}")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_panel(img_paths: list[Path], out_path: Path) -> None:
    circles = ["①", "②", "③", "④"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4), dpi=160)
    for ax, p, circ in zip(axes.flat, img_paths, circles):
        ax.imshow(imread(p))
        ax.set_title(circ, fontsize=14, pad=4)
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Numeric charts only. Secondary-row foils use wrong numbers (so rank≠audio).
ITEMS: dict[str, list[dict]] = {
    # Audio: 시간 45/40/15 + 목적 출근45>통학35>여가20 → answer ②
    "visual_match_003": [
        {
            "kind": "vbar",
            "title": "시간대별 이용률",
            "labels": ["오전 8시", "오후 6시", "그 외"],
            "values": [40, 45, 15],
        },
        {
            "kind": "vbar",
            "title": "시간대별 이용률",
            "labels": ["오전 8시", "오후 6시", "그 외"],
            "values": [45, 40, 15],  # CORRECT
        },
        {
            # foil: 통학>출근>여가
            "kind": "pie",
            "title": "이용 목적별 비율",
            "labels": ["출근", "통학", "여가·기타"],
            "values": [30, 45, 25],
        },
        {
            # foil: 여가>통학>출근
            "kind": "pie",
            "title": "이용 목적별 비율",
            "labels": ["출근", "통학", "여가·기타"],
            "values": [25, 30, 45],
        },
    ],
    # Audio: 연령 35/30/35 + 장르 코미디45>액션35>드라마20 → answer ①
    "visual_match_007": [
        {
            "kind": "pie",
            "title": "연령별 비율",
            "labels": ["20대", "30대", "40대 이상"],
            "values": [35, 30, 35],  # CORRECT
        },
        {
            "kind": "pie",
            "title": "연령별 비율",
            "labels": ["20대", "30대", "40대 이상"],
            "values": [30, 40, 30],
        },
        {
            # foil: 액션>코미디>드라마
            "kind": "hbar",
            "title": "선호 장르별 비율",
            "labels": ["코미디", "액션", "드라마"],
            "values": [30, 45, 25],
        },
        {
            # foil: 드라마>액션>코미디
            "kind": "hbar",
            "title": "선호 장르별 비율",
            "labels": ["코미디", "액션", "드라마"],
            "values": [20, 30, 50],
        },
    ],
    # Audio: 참여율 62/58/71 + 교육 횟수 C12>A9>B6 → answer ④
    "visual_match_012": [
        {
            # foil: A>C>B counts
            "kind": "hbar",
            "title": "분리수거 교육 횟수",
            "labels": ["A구", "B구", "C구"],
            "values": [12, 6, 9],
            "unit": "회",
            "xmax": 16,
        },
        {
            # foil: B>A>C
            "kind": "hbar",
            "title": "분리수거 교육 횟수",
            "labels": ["A구", "B구", "C구"],
            "values": [8, 12, 6],
            "unit": "회",
            "xmax": 16,
        },
        {
            "kind": "vbar",
            "title": "재활용 참여율",
            "labels": ["A구", "B구", "C구"],
            "values": [71, 58, 62],
        },
        {
            "kind": "vbar",
            "title": "재활용 참여율",
            "labels": ["A구", "B구", "C구"],
            "values": [62, 58, 71],  # CORRECT
        },
    ],
}


def main() -> None:
    font = setup_korean_font()
    print(f"font: {font}")
    print(f"out:  {OUT_ROOT}")

    for folder, options in ITEMS.items():
        assert options[0]["kind"] == options[1]["kind"]
        assert options[0]["title"] == options[1]["title"]
        assert options[2]["kind"] == options[3]["kind"]
        assert options[2]["title"] == options[3]["title"]
        assert options[0]["kind"] != options[2]["kind"]
        assert options[0]["title"] != options[2]["title"]

        paths: list[Path] = []
        for i, opt in enumerate(options, start=1):
            path = OUT_ROOT / folder / f"img{i}.png"
            render_option(opt, path)
            paths.append(path)
            print(
                f"  wrote {path.relative_to(ROOT)}  "
                f"[{opt['kind']}] {opt['title']}"
            )
        panel = OUT_ROOT / folder / "panel.png"
        make_panel(paths, panel)
        print(f"  wrote {panel.relative_to(ROOT)}")

    print("done.")


if __name__ == "__main__":
    main()
