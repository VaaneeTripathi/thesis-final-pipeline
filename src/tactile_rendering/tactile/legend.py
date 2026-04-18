"""Stage 5: Legend rendering.

Generates a separate SVG containing the numbered legend that maps
numeric IDs to braille labels. For refreshable pin displays, this
is the primary way users access node/edge text.
"""

from __future__ import annotations

import svgwrite

from tactile.braille import BrailleLabel

# Layout constants (mm).
ROW_HEIGHT_MM = 8.0
LEFT_MARGIN_MM = 5.0
BRAILLE_COL_MM = 25.0
FONT_SIZE_MM = 4.0
TOP_MARGIN_MM = 8.0
LEGEND_WIDTH_MM = 100.0


def render_legend(labels: list[BrailleLabel]) -> svgwrite.Drawing:
    """Render a legend SVG with numbered braille labels.

    Each row contains:  [number]  print_text  braille_text
    """
    num_rows = len(labels)
    canvas_h = TOP_MARGIN_MM + num_rows * ROW_HEIGHT_MM + TOP_MARGIN_MM

    dwg = svgwrite.Drawing(
        size=(f"{LEGEND_WIDTH_MM}mm", f"{canvas_h}mm"),
        viewBox=f"0 0 {LEGEND_WIDTH_MM} {canvas_h}",
    )

    # Title.
    dwg.add(dwg.text(
        "Legend",
        insert=(LEFT_MARGIN_MM, TOP_MARGIN_MM - 2),
        font_size=f"{FONT_SIZE_MM + 1}mm",
        font_family="sans-serif",
        font_weight="bold",
        fill="black",
    ))

    for i, bl in enumerate(labels):
        y = TOP_MARGIN_MM + (i + 1) * ROW_HEIGHT_MM

        # Number column.
        dwg.add(dwg.text(
            f"[{bl.legend_number}]",
            insert=(LEFT_MARGIN_MM, y),
            font_size=f"{FONT_SIZE_MM}mm",
            font_family="sans-serif",
            font_weight="bold",
            fill="black",
        ))

        # Print text column.
        dwg.add(dwg.text(
            bl.print_text,
            insert=(LEFT_MARGIN_MM + 15, y),
            font_size=f"{FONT_SIZE_MM}mm",
            font_family="sans-serif",
            fill="black",
        ))

        # Braille column.
        dwg.add(dwg.text(
            bl.braille_text,
            insert=(BRAILLE_COL_MM + 35, y),
            font_size=f"{FONT_SIZE_MM}mm",
            font_family="sans-serif",
            fill="black",
        ))

    return dwg
