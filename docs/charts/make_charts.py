"""Draw the README's charts from numbers measured on this project.

    python docs/charts/make_charts.py

Writes SVG files to docs/images/. Nothing here is estimated: each dataset
names what was measured, on what, and how to reproduce it. The SVGs switch
between light and dark ink with the viewer's colour scheme, so they read on
both GitHub themes.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images"

# Emphasis form: the series the chart is about in the accent, context in gray.
# Validated with the dataviz palette checks against GitHub's light (#ffffff)
# and dark (#0d1117) backgrounds: contrast >= 3:1, CVD separation >= 9.6.
STYLE = """
.title{font:600 15px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#1f2328}
.sub{font:400 12px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#59636e}
.label{font:400 13px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#1f2328}
.value{font:500 12px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#59636e;font-variant-numeric:tabular-nums}
.tick{font:400 11px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#818b98;font-variant-numeric:tabular-nums}
.grid{stroke:#d1d9e0;stroke-width:1}
.base{stroke:#9198a1;stroke-width:1}
.accent{fill:#d9541e}
.context{fill:#8b8a84}
@media (prefers-color-scheme: dark){
.title,.label{fill:#e6edf3}
.sub,.value{fill:#9198a1}
.tick{fill:#768390}
.grid{stroke:#30363d}
.base{stroke:#656c76}
.accent{fill:#e0602a}
.context{fill:#7d7c76}
}
"""

WIDTH = 720
LABEL_WIDTH = 176
RIGHT_PAD = 64
BAR = 16
PAIR_GAP = 2
GROUP_GAP = 18


@dataclass
class Series:
    name: str
    css: str
    values: list[float]


def _bar(x0: float, x1: float, y: float, height: float, css: str) -> str:
    """A horizontal bar, square at the baseline, 4px rounded at the data end."""
    radius = min(4.0, max(x1 - x0, 0) / 2, height / 2)
    if x1 - x0 < 0.5:
        return ""
    return (
        f'<path class="{css}" d="M{x0:.1f},{y:.1f} H{x1 - radius:.1f} '
        f"A{radius:.1f},{radius:.1f} 0 0 1 {x1:.1f},{y + radius:.1f} V{y + height - radius:.1f} "
        f'A{radius:.1f},{radius:.1f} 0 0 1 {x1 - radius:.1f},{y + height:.1f} H{x0:.1f} Z"/>'
    )


def bar_chart(
    name: str,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[Series],
    ticks: list[float],
    unit: str,
    fmt: str = "{:g}",
    note: str = "",
    tick_fmt: str | None = None,
) -> None:
    top = 58 + (22 if len(series) > 1 else 0)
    group = len(series) * BAR + (len(series) - 1) * PAIR_GAP
    plot_height = len(categories) * group + (len(categories) - 1) * GROUP_GAP
    height = top + plot_height + 34 + (18 if note else 0)
    x0 = LABEL_WIDTH
    x_max = WIDTH - RIGHT_PAD
    scale = (x_max - x0) / ticks[-1]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="t d">',
        f"<title id=\"t\">{escape(title)}</title>",
    ]
    rows = "; ".join(
        f"{category}: " + ", ".join(f"{s.name} {fmt.format(s.values[i])} {unit}" for s in series)
        for i, category in enumerate(categories)
    )
    parts.append(f'<desc id="d">{escape(subtitle)} {escape(rows)}</desc>')
    parts.append(f"<style>{STYLE}</style>")
    parts.append(f'<text class="title" x="0" y="20">{escape(title)}</text>')
    parts.append(f'<text class="sub" x="0" y="40">{escape(subtitle)}</text>')

    if len(series) > 1:
        x = 0.0
        for item in series:
            parts.append(f'<rect class="{item.css}" x="{x:.1f}" y="55" width="10" height="10" rx="2"/>')
            parts.append(f'<text class="sub" x="{x + 16:.1f}" y="64">{escape(item.name)}</text>')
            x += 16 + 7.2 * len(item.name) + 22

    bottom = top + plot_height
    for tick in ticks:
        tx = x0 + tick * scale
        parts.append(f'<line class="grid" x1="{tx:.1f}" y1="{top - 6}" x2="{tx:.1f}" y2="{bottom + 4}"/>')
        parts.append(
            f'<text class="tick" x="{tx:.1f}" y="{bottom + 20}" text-anchor="middle">{(tick_fmt or fmt).format(tick)}</text>'
        )
    parts.append(f'<line class="base" x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{bottom + 4}"/>')

    y = float(top)
    for index, category in enumerate(categories):
        parts.append(
            f'<text class="label" x="{x0 - 12}" y="{y + group / 2 + 4.5:.1f}" text-anchor="end">{escape(category)}</text>'
        )
        for item in series:
            value = item.values[index]
            x1 = x0 + value * scale
            parts.append(_bar(x0, x1, y, BAR, item.css))
            parts.append(
                f'<text class="value" x="{x1 + 6:.1f}" y="{y + BAR / 2 + 4:.1f}">{fmt.format(value)} {escape(unit)}</text>'
            )
            y += BAR + PAIR_GAP
        y += GROUP_GAP - PAIR_GAP

    if note:
        parts.append(f'<text class="tick" x="0" y="{height - 4}">{escape(note)}</text>')
    parts.append("</svg>")
    (OUT / f"{name}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote docs/images/{name}.svg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # From `python -m benchmarks.run`: 9 benchmark inputs, 3 of each type.
    # Without routing every one of the 7 capabilities runs on every input.
    bar_chart(
        "chart-routing",
        "Models invoked per input",
        "Routing benchmark: 21 invocations with routing, 63 without, 66.7% fewer.",
        ["Image", "Document", "Text"],
        [
            Series("With routing", "accent", [4, 1, 1]),
            Series("Every model on every input", "context", [7, 7, 7]),
        ],
        ticks=[0, 1, 2, 3, 4, 5, 6, 7],
        unit="",
        note="Source: python -m benchmarks.run (deterministic, no models needed).",
    )

    # Measured in the paper-profile container limited to 2 CPUs (docker --cpus=2),
    # warm model, same input for both runs. BLIP-large: 8-beam caption of COCO
    # val2017 image 872, one warm call each. T5-base: 4-beam summary of the same
    # six-paragraph text, mean of two warm calls each (see FP32_T5 below).
    bar_chart(
        "chart-int8",
        "Seconds per call on 2 CPU cores",
        "Same input and settings. T5's summary was identical; BLIP's caption was worded differently.",
        ["BLIP-large caption", "T5-base summary"],
        [
            Series("int8 (as shipped)", "accent", [7.3, INT8_T5]),
            Series("fp32", "context", [48.2, FP32_T5]),
        ],
        ticks=[0, 10, 20, 30, 40, 50],
        unit="s",
        fmt="{:g}",
        note="Measured with docker --cpus=2 on the project's image, warm model (T5: mean of two calls).",
    )

    # PaddleOCR PP-OCRv5 (server detector, English recogniser, oneDNN on),
    # container limited to 2 CPUs. Default: PaddleOCR's own 8 threads.
    bar_chart(
        "chart-ocr-threads",
        "PaddleOCR seconds per image on 2 CPU cores",
        "Thread count matched to the container's CPU quota, against PaddleOCR's default of 8.",
        ["Printed notice", "Scanned page", "Street photo"],
        [
            Series("Threads = CPU quota (as shipped)", "accent", [4.3, 5.7, 2.0]),
            Series("Default threads", "context", [8.2, 12.1, 4.1]),
        ],
        ticks=[0, 2, 4, 6, 8, 10, 12, 14],
        unit="s",
        fmt="{:.1f}",
        tick_fmt="{:g}",
        note="Measured with docker --cpus=2; one run per image, in the same order for both settings.",
    )

    # File sizes of the weights fetched from the Hugging Face Hub (repository
    # listings). PaddleOCR is the server detector, English, Telugu and
    # Devanagari recognisers and the line-orientation model together.
    sizes = [
        ("NLLB-200 (600M)", 2460.5), ("BLIP (large)", 1879.0), ("T5 (base)", 891.6),
        ("RoBERTa sentiment", 501.0), ("YOLOv8x", 136.9), ("PaddleOCR PP-OCRv5", 118.1),
        ("ResNet50", 102.5), ("spaCy en_core_web_sm", 12.8), ("YOLOv8n", 6.5),
    ]
    bar_chart(
        "chart-weights",
        "Model weights on disk, MB",
        f"The nine paper models, {sum(size for _, size in sizes) / 1000:.1f} GB in all, fetched from the Hugging Face Hub.",
        [label for label, _ in sizes],
        [Series("Weights", "accent", [size for _, size in sizes])],
        ticks=[0, 500, 1000, 1500, 2000, 2500],
        unit="MB",
        fmt="{:,.1f}",
        tick_fmt="{:,.0f}",
    )


# T5-base, same text, mean of two warm calls each after a discarded warm-up:
# fp32 10.48 s and 9.21 s, int8 4.02 s and 4.26 s. Identical output text.
FP32_T5 = 9.8
INT8_T5 = 4.1

if __name__ == "__main__":
    main()
