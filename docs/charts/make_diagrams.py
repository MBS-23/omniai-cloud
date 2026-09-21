"""Draw the README's architecture and processing-flow diagrams.

    python docs/charts/make_diagrams.py

Writes SVG files to docs/images/. The boxes name real modules and real
decisions in backend/core/pipeline.py and backend/agent/, so when those
change, change this file with them.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images"

STYLE = """
text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
.h{font-size:13px;font-weight:600;fill:#1f2328}
.t{font-size:12px;fill:#1f2328}
.m{font-size:11px;fill:#59636e}
.lane{font-size:12px;font-weight:600;fill:#59636e;letter-spacing:.04em}
.box{fill:#f6f8fa;stroke:#d1d9e0;stroke-width:1.25}
.key{fill:#fff4ef;stroke:#d9541e;stroke-width:1.25}
.ask{fill:#ffffff;stroke:#d9541e;stroke-width:1.25;stroke-dasharray:4 3}
.layer{fill:none;stroke:#d1d9e0;stroke-width:1}
.wire{stroke:#818b98;stroke-width:1.4;fill:none}
.tip{fill:#818b98}
.yes{font-size:11px;font-weight:600;fill:#d9541e}
@media (prefers-color-scheme: dark){
.h,.t{fill:#e6edf3}
.m,.lane{fill:#9198a1}
.box{fill:#161b22;stroke:#3d444d}
.key{fill:#2a1810;stroke:#e0602a}
.ask{fill:#0d1117;stroke:#e0602a}
.layer{stroke:#3d444d}
.wire{stroke:#768390}
.tip{fill:#768390}
.yes{fill:#e0602a}
}
"""


class Canvas:
    def __init__(self, width: int, height: int, title: str, desc: str) -> None:
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
            f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>',
            f"<style>{STYLE}</style>",
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse"><path class="tip" d="M0,0 L10,5 L0,10 z"/></marker></defs>',
        ]

    def box(self, x, y, w, h, heading, lines=(), css="box", align="middle"):
        self.parts.append(f'<rect class="{css}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        cx = x + w / 2 if align == "middle" else x + 12
        total = 16 + 15 * len(lines)
        ty = y + h / 2 - total / 2 + 12
        self.parts.append(f'<text class="h" x="{cx}" y="{ty:.1f}" text-anchor="{align if align == "middle" else "start"}">{escape(heading)}</text>')
        for index, line in enumerate(lines):
            self.parts.append(
                f'<text class="m" x="{cx}" y="{ty + 16 + 15 * index:.1f}" '
                f'text-anchor="{align if align == "middle" else "start"}">{escape(line)}</text>'
            )

    def wire(self, points, label=None, label_at=None, css="yes"):
        path = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(points))
        self.parts.append(f'<path class="wire" d="{path}" marker-end="url(#a)"/>')
        if label:
            lx, ly = label_at or points[0]
            self.parts.append(f'<text class="{css}" x="{lx}" y="{ly}">{escape(label)}</text>')

    def text(self, x, y, value, css="t", anchor="start"):
        self.parts.append(f'<text class="{css}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(value)}</text>')

    def raw(self, markup):
        self.parts.append(markup)

    def save(self, name):
        self.parts.append("</svg>")
        (OUT / f"{name}.svg").write_text("\n".join(self.parts) + "\n", encoding="utf-8")
        print(f"wrote docs/images/{name}.svg")


def architecture() -> None:
    c = Canvas(
        900, 560,
        "OmniAI Cloud architecture",
        "Three layers. The interface layer serves the page and streams progress. The intelligence "
        "layer inspects the input, selects models, orchestrates and explains. The execution layer "
        "runs the paper's nine models locally; their weights come from the Hugging Face Hub once, "
        "at build time.",
    )
    # Browser
    c.box(300, 12, 300, 54, "Browser", ["Upload or text in, live progress and results out"])
    c.wire([(450, 66), (450, 92)])
    # Interface layer
    c.raw('<rect class="layer" x="10" y="92" width="880" height="96" rx="12"/>')
    c.text(26, 114, "INTERFACE LAYER", "lane")
    c.box(40, 124, 250, 52, "Page and static assets", ["routes/views.py, templates, static"])
    c.box(325, 124, 250, 52, "JSON and streaming API", ["routes/api.py, request checks"])
    c.box(610, 124, 250, 52, "Progress stream (SSE)", ["one event per stage, as it happens"])
    c.wire([(450, 188), (450, 214)])
    # Intelligence layer
    c.raw('<rect class="layer" x="10" y="214" width="880" height="118" rx="12"/>')
    c.text(26, 236, "INTELLIGENCE LAYER", "lane")
    steps = [
        ("Inspect", ["what arrived"]),
        ("Auto-selector", ["which models, why"]),
        ("Orchestrate", ["run, skip, re-route"]),
        ("Refine labels", ["cat to lion"]),
        ("Explain", ["every decision"]),
    ]
    x = 40
    for index, (heading, lines) in enumerate(steps):
        c.box(x, 250, 150, 62, heading, lines, css="key" if index in (1, 3) else "box")
        if index < len(steps) - 1:
            c.wire([(x + 150, 281), (x + 170, 281)])
        x += 170
    c.wire([(450, 332), (450, 358)])
    # Execution layer
    c.raw('<rect class="layer" x="10" y="358" width="880" height="126" rx="12"/>')
    c.text(26, 380, "EXECUTION LAYER: THE PAPER'S MODELS, RUNNING LOCALLY", "lane")
    models = [
        ("ResNet50", "classify"), ("YOLOv8n", "detect"), ("YOLOv8x", "wildlife"),
        ("PaddleOCR", "read text"), ("BLIP-large", "caption"), ("NLLB-200", "translate"),
        ("RoBERTa", "sentiment"), ("spaCy", "entities"), ("T5", "summarise"),
    ]
    x = 40
    for heading, role in models:
        c.box(x, 396, 88, 62, heading, [role])
        x += 92.5
    c.text(450, 476, "Loaded once, pre-loaded in the background; int8 on CPU for BLIP, NLLB and T5", "m", "middle")
    c.wire([(450, 484), (450, 510)])
    # Hub
    c.box(250, 510, 400, 44, "Hugging Face Hub", ["weights fetched once while the image builds; none at request time"])
    c.save("architecture")


def flow() -> None:
    c = Canvas(
        900, 700,
        "How an input is routed",
        "Images are classified first. A wildlife label skips OCR and switches detection from YOLOv8n "
        "to YOLOv8x, then refines the detector's coarse label. Otherwise OCR runs; if text covers ten "
        "percent or more of the image it is treated as a page and scene analysis is skipped. Documents "
        "use their text layer or OCR, then T5 summarises. Text is translated when needed, then spaCy "
        "and RoBERTa analyse it. Every path ends in an explanation.",
    )
    # --- image lane ---
    c.text(20, 26, "IMAGE", "lane")
    c.box(20, 38, 170, 56, "ResNet50", ["classify the subject"], css="box")
    c.wire([(190, 66), (230, 66)])
    c.box(230, 38, 200, 56, "Wildlife name?", ["791-name list, label from ResNet50"], css="ask")
    c.wire([(430, 66), (480, 66)], "yes", (440, 58))
    c.box(480, 38, 170, 56, "YOLOv8x", ["low threshold, all animals"], css="key")
    c.wire([(650, 66), (690, 66)])
    c.box(690, 38, 190, 56, "Refine + caption", ["cat to lion; BLIP with a hint"], css="key")
    c.wire([(330, 94), (330, 132)], "no", (338, 118))
    c.box(230, 132, 200, 56, "PaddleOCR", ["English, then Telugu, then Hindi"], css="box")
    c.wire([(430, 160), (480, 160)])
    c.box(480, 132, 170, 56, "Text covers 10%+?", ["of the image area"], css="ask")
    c.wire([(650, 160), (690, 160)], "yes", (656, 152))
    c.box(690, 132, 190, 56, "A page, not a scene", ["keep the text, skip the rest"], css="box")
    c.wire([(565, 188), (565, 226)], "no", (573, 212))
    c.box(480, 226, 170, 56, "YOLOv8n", ["COCO objects, with boxes"], css="box")
    c.wire([(650, 254), (690, 254)])
    c.box(690, 226, 190, 56, "BLIP-large", ["one-sentence description"], css="box")

    # --- document lane ---
    c.raw('<line class="layer" x1="10" y1="310" x2="890" y2="310"/>')
    c.text(20, 340, "DOCUMENT (PDF, DOCX)", "lane")
    c.box(20, 352, 170, 56, "Text layer?", ["PDF with real text, or DOCX"], css="ask")
    c.wire([(190, 380), (230, 380)], "yes", (196, 372))
    c.box(230, 352, 200, 56, "Extract the text", ["no OCR needed"], css="box")
    c.wire([(105, 408), (105, 446)], "no", (113, 432))
    c.box(20, 446, 170, 56, "Render pages", ["first 15, at 200 DPI"], css="box")
    c.wire([(190, 474), (230, 474)])
    c.box(230, 446, 200, 56, "PaddleOCR", ["page by page"], css="box")
    c.wire([(430, 380), (455, 380), (455, 418), (480, 418)])
    c.wire([(430, 474), (455, 474), (455, 434), (480, 434)])
    c.box(480, 398, 170, 56, "T5 summary", ["spread across the document"], css="key")
    c.wire([(650, 426), (690, 426)])
    c.box(690, 398, 190, 56, "Text + summary", ["and how it was read"], css="box")

    # --- text lane ---
    c.raw('<line class="layer" x1="10" y1="528" x2="890" y2="528"/>')
    c.text(20, 558, "TEXT", "lane")
    c.box(20, 570, 170, 56, "langdetect", ["which language"], css="box")
    c.wire([(190, 598), (230, 598)])
    c.box(230, 570, 200, 56, "NLLB-200", ["translate if the target differs"], css="box")
    c.wire([(430, 598), (480, 598)])
    c.box(480, 570, 170, 56, "spaCy + RoBERTa", ["entities, sentiment"], css="key")
    c.wire([(650, 598), (690, 598)])
    c.box(690, 570, 190, 56, "Findings", ["in English, with the reasoning"], css="box")
    c.text(450, 676, "Every path ends in the same place: a plain-language account of what ran, what was skipped and why.", "m", "middle")
    c.save("flow")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    flow()
