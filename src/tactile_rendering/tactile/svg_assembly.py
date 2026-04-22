"""Stage 4: Structural SVG assembly — diagram only (not the key).

Produces a tactile-ready SVG of the flowchart with:
  - BANA-compliant line weights and node shapes
  - A bold numeric legend ID centred inside each node
  - Filled arrowheads on directed edges
  - Clean canvas margins

The key (label lookup) is produced separately in stage 5 (legend.py).

BANA line-weight rule: minimum 0.5 mm.  1.5 mm is recommended for
embossing and is the value used here.
"""

from __future__ import annotations

import math
from pathlib import Path

import svgwrite

from tactile.ir import Flowchart
from tactile.layout import LayoutResult, NodeLayout, EdgeLayout
from tactile.braille import transcribe_labels, BrailleLabel

INCHES_TO_MM = 25.4

# Line weight: BANA minimum is 0.5 mm; 1.5 mm suits embossing and the
# 60×40 pin grid (a 1 mm line covers less than half a pin at 2.4 mm pitch).
STROKE_WIDTH_MM = 1.5

# Dashed/dotted line segments (BANA: dash 6–10 mm, gap ≈ half dash).
DASH_MM  = 7.0
GAP_MM   = 3.5

# Arrowhead: filled solid triangle, 5 mm base.
ARROW_SIZE_MM = 5.0

# Number inside node: 5 mm keeps the digit from visually dominating the outline.
# Note: font-size is specified as a bare user-unit number (not "Xmm") so that
# CairoSVG's 72-DPI mm resolver doesn't produce a different size than the
# 300-DPI viewBox scale used for rasterisation.
NODE_NUM_FONT_MM = 5.0

# Canvas margin around the graph bounding box.
CANVAS_MARGIN_MM = 15.0


def _mm(inches: float) -> float:
    return inches * INCHES_TO_MM


def _stroke_attrs(line_type: str = "solid") -> dict:
    base = {"stroke": "black", "stroke_width": STROKE_WIDTH_MM, "fill": "none"}
    if line_type == "dashed":
        base["stroke_dasharray"] = f"{DASH_MM},{GAP_MM}"
    elif line_type == "dotted":
        base["stroke_dasharray"] = f"{STROKE_WIDTH_MM},{GAP_MM}"
    return base


def _draw_node_shape(dwg: svgwrite.Drawing, nl: NodeLayout, legend_num: int) -> None:
    """Draw the node outline and its numeric ID centred inside."""
    cx  = _mm(nl.cx)
    cy  = _mm(nl.cy)
    w   = _mm(nl.width)
    h   = _mm(nl.height)
    hw  = w / 2
    hh  = h / 2
    sa  = _stroke_attrs()

    shape = nl.shape

    if shape in ("rectangle", "other"):
        dwg.add(dwg.rect(insert=(cx - hw, cy - hh), size=(w, h), **sa))

    elif shape == "rounded-rectangle":
        corner = min(w, h) * 0.18
        dwg.add(dwg.rect(insert=(cx - hw, cy - hh), size=(w, h),
                         rx=corner, ry=corner, **sa))

    elif shape == "diamond":
        pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
        dwg.add(dwg.polygon(pts, **sa))

    elif shape in ("oval", "ellipse"):
        dwg.add(dwg.ellipse(center=(cx, cy), r=(hw, hh), **sa))

    elif shape == "circle":
        dwg.add(dwg.circle(center=(cx, cy), r=min(hw, hh), **sa))

    elif shape == "parallelogram":
        slant = hw * 0.25
        pts = [
            (cx - hw + slant, cy - hh), (cx + hw,         cy - hh),
            (cx + hw - slant, cy + hh), (cx - hw,         cy + hh),
        ]
        dwg.add(dwg.polygon(pts, **sa))

    elif shape == "triangle":
        pts = [(cx, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
        dwg.add(dwg.polygon(pts, **sa))

    else:
        dwg.add(dwg.rect(insert=(cx - hw, cy - hh), size=(w, h), **sa))

    # Numeric ID — centred in the node.
    # y is set so the cap-height midpoint lands on the node centre.
    # cap_height ≈ 0.7 × font_size; we shift baseline down by half that.
    # font_size is in user units (= mm) with no "mm" suffix to avoid
    # CairoSVG resolving mm at 72 DPI independently of the raster scale.
    text_y = cy + NODE_NUM_FONT_MM * 0.35
    dwg.add(dwg.text(
        str(legend_num),
        insert=(cx, text_y),
        text_anchor="middle",
        font_size=str(NODE_NUM_FONT_MM),
        font_family="Helvetica, Arial, sans-serif",
        font_weight="bold",
        fill="black",
    ))


def _draw_arrowhead(dwg: svgwrite.Drawing,
                    tip_x: float, tip_y: float,
                    from_x: float, from_y: float) -> None:
    """Filled triangular arrowhead at (tip_x, tip_y) pointing away from (from_x, from_y)."""
    dx, dy = tip_x - from_x, tip_y - from_y
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length   # unit vector along edge
    px, py = -uy, ux                    # perpendicular

    base_x = tip_x - ux * ARROW_SIZE_MM
    base_y = tip_y - uy * ARROW_SIZE_MM
    half   = ARROW_SIZE_MM * 0.45

    pts = [
        (tip_x, tip_y),
        (base_x + px * half, base_y + py * half),
        (base_x - px * half, base_y - py * half),
    ]
    dwg.add(dwg.polygon(pts, fill="black", stroke="none"))


def _draw_edge(dwg: svgwrite.Drawing, el: EdgeLayout, line_type: str = "solid") -> None:
    if len(el.points) < 2:
        return

    pts_mm = [(_mm(x), _mm(y)) for x, y in el.points]

    dwg.add(dwg.polyline(pts_mm, **_stroke_attrs(line_type)))

    if el.direction in ("forward", "bidirectional"):
        _draw_arrowhead(dwg, pts_mm[-1][0], pts_mm[-1][1],
                        pts_mm[-2][0], pts_mm[-2][1])
    if el.direction in ("backward", "bidirectional"):
        _draw_arrowhead(dwg, pts_mm[0][0], pts_mm[0][1],
                        pts_mm[1][0], pts_mm[1][1])


def assemble(
    flowchart: Flowchart,
    layout: LayoutResult,
) -> tuple[svgwrite.Drawing, list[BrailleLabel], dict]:
    """Assemble the diagram SVG, braille labels, and sidecar metadata.

    Returns:
        (svg_drawing, braille_labels, sidecar_dict)
    """
    # Build legend mapping: node order → sequential number starting at 1.
    label_pairs = [(n.id, n.text or n.id) for n in flowchart.nodes]
    braille_labels = transcribe_labels(label_pairs)

    # Connection labels continue numbering after nodes.
    conn_label_pairs = [(c.id, c.label) for c in flowchart.connections if c.label]
    if conn_label_pairs:
        conn_braille = transcribe_labels(conn_label_pairs)
        offset = len(braille_labels)
        for i, bl in enumerate(conn_braille):
            bl.legend_number = offset + i + 1
        braille_labels.extend(conn_braille)

    # Build connection line-type lookup.
    conn_line_type: dict[str, str] = {
        c.id: c.line_type for c in flowchart.connections
    }

    # SVG canvas.
    m = CANVAS_MARGIN_MM
    canvas_w = _mm(layout.graph_width)  + 2 * m
    canvas_h = _mm(layout.graph_height) + 2 * m

    dwg = svgwrite.Drawing(
        size=(f"{canvas_w}mm", f"{canvas_h}mm"),
        viewBox=f"{-m} {-m} {canvas_w} {canvas_h}",
    )

    legend_lookup = {
        label_pairs[i][0]: braille_labels[i].legend_number
        for i in range(len(label_pairs))
    }

    # Edges behind nodes.
    for edge in layout.edges:
        lt = conn_line_type.get(edge.id, "solid")
        _draw_edge(dwg, edge, lt)

    # Nodes on top.
    for node_id, nl in layout.nodes.items():
        _draw_node_shape(dwg, nl, legend_lookup.get(node_id, 0))

    sidecar = {
        "source_ir_id": flowchart.id,
        "label_strategy": "key",
        "bana_conformance": {
            "stroke_width_mm":   STROKE_WIDTH_MM,
            "arrow_size_mm":     ARROW_SIZE_MM,
            "node_num_font_mm":  NODE_NUM_FONT_MM,
            "canvas_margin_mm":  CANVAS_MARGIN_MM,
        },
        "legend": [
            {
                "number":      bl.legend_number,
                "print_text":  bl.print_text,
                "braille_text": bl.braille_text,
            }
            for bl in braille_labels
        ],
    }

    return dwg, braille_labels, sidecar
