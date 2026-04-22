"""Stage 5: Key rendering

Produces a separate SVG intended for a tactile embosser. The key maps numeric IDs (shown inside nodes on the
diagram) to their print labels and braille equivalents.

Braille is rendered as explicit SVG circles — NOT as Unicode text.
This eliminates all font-metric variability (different systems render the
U+2800–U+28FF block at wildly different sizes and baselines). Each dot is
a circle of known radius at a calculated coordinate.

Standard braille physical dimensions used here:
  - Dot radius:            0.75 mm  (diameter 1.5 mm per BANA spec)
  - Dot pitch (x and y):  2.3  mm  (centre-to-centre within a cell)
  - Cell spacing:          6.2  mm  (centre of left column, cell to cell)

BANA rules implemented:
  - Title is "Key:" — BANA explicitly prohibits "Legend"
  - Each entry: print label row, then braille dot row below it
  - Thick outer border frames the whole key
  - Thin dashed separator between entries
  - Canvas 165 mm wide, 12 mm margins
"""

from __future__ import annotations

import svgwrite

from tactile.braille import BrailleLabel

CANVAS_W_MM  = 165.0
OUTER_MARGIN = 12.0

TITLE_FONT   = 6.0          # user units (= mm in this coordinate system)
TITLE_H_MM   = 18.0         # height of title area including thick rule


PRINT_ROW_H   = 9.0         # height for the print-text row
BRAILLE_ROW_H = 14.0        # height for the braille-dot row
                             # (3 dots × 2.3 mm pitch + 2 × 1.5 mm padding ≈ 10 mm;
                             #  14 mm gives 2 mm top-pad + 6.9 mm dots + 5.1 mm bot-pad)
ENTRY_GAP     = 5.0         # gap between entries (includes separator rule)
ENTRY_H       = PRINT_ROW_H + BRAILLE_ROW_H   # 23 mm per entry


NUM_X   = OUTER_MARGIN               # number column
TEXT_X  = OUTER_MARGIN + 14.0        # print label + braille start


PRINT_FONT = 5.0    # user units
NUM_FONT   = 5.0    # user units

BORDER_STROKE = 1.5
RULE_STROKE   = 0.4

DOT_RADIUS    = 0.75   # mm
DOT_PITCH     = 2.3    # mm centre-to-centre within a cell (horizontal and vertical)
CELL_SPACING  = 6.2    # mm centre of col-0, cell to cell

# UEB / Grade-1 Unicode braille bit layout (U+2800 + bits):
# Bit 0 (0x01) = dot 1 → col 0, row 0  (top-left)
# Bit 1 (0x02) = dot 2 → col 0, row 1  (mid-left)
# Bit 2 (0x04) = dot 3 → col 0, row 2  (bot-left)
# Bit 3 (0x08) = dot 4 → col 1, row 0  (top-right)
# Bit 4 (0x10) = dot 5 → col 1, row 1  (mid-right)
# Bit 5 (0x20) = dot 6 → col 1, row 2  (bot-right)
_DOT_BIT: list[tuple[int, int]] = [
    (0, 0),  # bit 0 → dot 1
    (0, 1),  # bit 1 → dot 2
    (0, 2),  # bit 2 → dot 3
    (1, 0),  # bit 3 → dot 4
    (1, 1),  # bit 4 → dot 5
    (1, 2),  # bit 5 → dot 6
]

MAX_PER_PAGE = 20

def _draw_braille_dots(
    dwg: svgwrite.Drawing,
    braille_text: str,
    x_start: float,
    y_top: float,
) -> None:
    """Draw braille dots as SVG circles for the given braille string.

    Args:
        braille_text: String of Unicode braille characters (U+2800–U+28FF).
        x_start:      Left x position for the first cell's left column.
        y_top:        Top y of the braille row area (dots are centred within
                      BRAILLE_ROW_H below this).
    """
    # Centre the 3-row dot grid vertically within BRAILLE_ROW_H.
    # Total dot span height = 2 * DOT_PITCH + 2 * DOT_RADIUS = 2*2.3 + 1.5 = 6.1 mm
    dot_span_h = 2 * DOT_PITCH + 2 * DOT_RADIUS
    y_dot_top = y_top + (BRAILLE_ROW_H - dot_span_h) / 2 + DOT_RADIUS

    for cell_idx, ch in enumerate(braille_text):
        cp = ord(ch)
        if cp < 0x2800 or cp > 0x28FF:
            continue   # skip non-braille characters
        bits = cp - 0x2800
        if bits == 0:
            continue   # empty cell — no dots to draw

        cell_x = x_start + cell_idx * CELL_SPACING

        for bit_pos, (col, row) in enumerate(_DOT_BIT):
            if bits & (1 << bit_pos):
                dot_cx = cell_x + col * DOT_PITCH
                dot_cy = y_dot_top + row * DOT_PITCH
                dwg.add(dwg.circle(
                    center=(dot_cx, dot_cy),
                    r=DOT_RADIUS,
                    fill="black",
                    stroke="none",
                ))


def _canvas_height(n: int) -> float:
    return TITLE_H_MM + n * ENTRY_H + max(0, n - 1) * ENTRY_GAP + 2 * OUTER_MARGIN


def _render_page(labels: list[BrailleLabel]) -> svgwrite.Drawing:
    n = len(labels)
    h = _canvas_height(n)

    dwg = svgwrite.Drawing(
        size=(f"{CANVAS_W_MM}mm", f"{h}mm"),
        viewBox=f"0 0 {CANVAS_W_MM} {h}",
    )

    # Outer border
    dwg.add(dwg.rect(
        insert=(BORDER_STROKE / 2, BORDER_STROKE / 2),
        size=(CANVAS_W_MM - BORDER_STROKE, h - BORDER_STROKE),
        stroke="black", stroke_width=BORDER_STROKE, fill="none",
    ))

    # Title
    title_baseline = OUTER_MARGIN + TITLE_FONT
    dwg.add(dwg.text(
        "Key:",
        insert=(OUTER_MARGIN, title_baseline),
        font_size=str(TITLE_FONT),
        font_family="Helvetica, Arial, sans-serif",
        font_weight="bold",
        fill="black",
    ))

    # Thick rule below title
    rule_y = OUTER_MARGIN + TITLE_H_MM - 3.0
    dwg.add(dwg.line(
        start=(OUTER_MARGIN, rule_y),
        end=(CANVAS_W_MM - OUTER_MARGIN, rule_y),
        stroke="black", stroke_width=BORDER_STROKE,
    ))

    content_top = OUTER_MARGIN + TITLE_H_MM

    for i, bl in enumerate(labels):
        entry_top = content_top + i * (ENTRY_H + ENTRY_GAP)

        print_baseline = entry_top + PRINT_ROW_H * 0.72

        dwg.add(dwg.text(
            f"{bl.legend_number}.",
            insert=(NUM_X, print_baseline),
            font_size=str(NUM_FONT),
            font_family="Helvetica, Arial, sans-serif",
            font_weight="bold",
            fill="black",
        ))
        dwg.add(dwg.text(
            bl.print_text or "",
            insert=(TEXT_X, print_baseline),
            font_size=str(PRINT_FONT),
            font_family="Helvetica, Arial, sans-serif",
            fill="black",
        ))

        braille_row_top = entry_top + PRINT_ROW_H
        if bl.braille_text:
            _draw_braille_dots(dwg, bl.braille_text, TEXT_X, braille_row_top)

        if i < n - 1:
            sep_y = entry_top + ENTRY_H + ENTRY_GAP / 2
            dwg.add(dwg.line(
                start=(OUTER_MARGIN, sep_y),
                end=(CANVAS_W_MM - OUTER_MARGIN, sep_y),
                stroke="black", stroke_width=RULE_STROKE,
                stroke_dasharray="2,2",
            ))

    return dwg

def render_legend(labels: list[BrailleLabel]) -> svgwrite.Drawing:
    """Return the first key page (backward-compatible single-page API)."""
    return _render_page(labels[:MAX_PER_PAGE])


def render_legend_pages(labels: list[BrailleLabel]) -> list[svgwrite.Drawing]:
    """Return a list of key page drawings, MAX_PER_PAGE entries each."""
    pages = []
    for start in range(0, max(1, len(labels)), MAX_PER_PAGE):
        pages.append(_render_page(labels[start: start + MAX_PER_PAGE]))
    return pages
