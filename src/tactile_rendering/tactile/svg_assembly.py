"""Stage 4: Structural SVG assembly.

Takes a LayoutResult and a Flowchart and emits:
  1. A tactile-ready SVG with BANA-aligned line weights, node shapes,
     routed edges, arrowheads, and numeric legend IDs inside nodes.
  2. A sidecar JSON with the legend table, braille translations, and
     BANA conformance metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import svgwrite

from tactile.ir import Flowchart
from tactile.layout import LayoutResult, NodeLayout, EdgeLayout
from tactile.braille import transcribe_labels, BrailleLabel

# --- Constants ---

# Conversion: Graphviz uses inches; we convert to mm for BANA alignment.
INCHES_TO_MM = 25.4

# BANA minimum line weight is 0.5 mm; we use 1 mm for better tactile discrimination.
STROKE_WIDTH_MM = 1.0

# Arrow head size in mm.
ARROW_SIZE_MM = 3.0

# Font size for numeric IDs inside nodes (in mm).
LABEL_FONT_MM = 4.0


def _inches_to_mm(val: float) -> float:
    return val * INCHES_TO_MM


def _draw_node_shape(
    dwg: svgwrite.Drawing,
    nl: NodeLayout,
    legend_num: int,
) -> None:
    """Draw a node shape and its numeric legend ID."""
    cx = _inches_to_mm(nl.cx)
    cy = _inches_to_mm(nl.cy)
    w = _inches_to_mm(nl.width)
    h = _inches_to_mm(nl.height)
    half_w = w / 2
    half_h = h / 2

    stroke_attrs = {
        "stroke": "black",
        "stroke_width": STROKE_WIDTH_MM,
        "fill": "none",
    }

    if nl.shape == "rectangle" or nl.shape == "other":
        dwg.add(dwg.rect(
            insert=(cx - half_w, cy - half_h),
            size=(w, h),
            **stroke_attrs,
        ))

    elif nl.shape == "rounded-rectangle":
        corner = min(w, h) * 0.2
        dwg.add(dwg.rect(
            insert=(cx - half_w, cy - half_h),
            size=(w, h),
            rx=corner, ry=corner,
            **stroke_attrs,
        ))

    elif nl.shape == "diamond":
        points = [
            (cx, cy - half_h),
            (cx + half_w, cy),
            (cx, cy + half_h),
            (cx - half_w, cy),
        ]
        dwg.add(dwg.polygon(points, **stroke_attrs))

    elif nl.shape in ("oval", "ellipse"):
        dwg.add(dwg.ellipse(
            center=(cx, cy),
            r=(half_w, half_h),
            **stroke_attrs,
        ))

    elif nl.shape == "circle":
        r = min(half_w, half_h)
        dwg.add(dwg.circle(center=(cx, cy), r=r, **stroke_attrs))

    elif nl.shape == "parallelogram":
        slant = half_w * 0.25
        points = [
            (cx - half_w + slant, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w - slant, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        dwg.add(dwg.polygon(points, **stroke_attrs))

    elif nl.shape == "triangle":
        points = [
            (cx, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        dwg.add(dwg.polygon(points, **stroke_attrs))

    else:
        # Fallback to rectangle for unknown shapes.
        dwg.add(dwg.rect(
            insert=(cx - half_w, cy - half_h),
            size=(w, h),
            **stroke_attrs,
        ))

    # Numeric legend ID centered in node.
    # At 60×40 rasterization a digit needs ~3 pins tall to be legible,
    # which requires ~3mm font-size given typical pin spacing. Non-bold
    # keeps the strokes from merging with the box outline at low res.
    font_mm = min(h * 0.35, w * 0.2)
    dwg.add(dwg.text(
        str(legend_num),
        insert=(cx, cy),
        text_anchor="middle",
        dominant_baseline="central",
        font_size=f"{font_mm:.2f}mm",
        font_family="sans-serif",
        fill="black",
    ))


def _draw_arrowhead(
    dwg: svgwrite.Drawing,
    tip_x: float,
    tip_y: float,
    from_x: float,
    from_y: float,
) -> None:
    """Draw a filled triangular arrowhead at (tip_x, tip_y)."""
    dx = tip_x - from_x
    dy = tip_y - from_y
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return

    # Unit vector along the edge direction.
    ux, uy = dx / length, dy / length
    # Perpendicular.
    px, py = -uy, ux

    size = ARROW_SIZE_MM
    base_x = tip_x - ux * size
    base_y = tip_y - uy * size
    half = size * 0.4

    points = [
        (tip_x, tip_y),
        (base_x + px * half, base_y + py * half),
        (base_x - px * half, base_y - py * half),
    ]
    dwg.add(dwg.polygon(points, fill="black", stroke="none"))


def _draw_edge(dwg: svgwrite.Drawing, el: EdgeLayout) -> None:
    """Draw an edge as a polyline with optional arrowheads."""
    if len(el.points) < 2:
        return

    mm_points = [(_inches_to_mm(x), _inches_to_mm(y)) for x, y in el.points]

    dwg.add(dwg.polyline(
        mm_points,
        stroke="black",
        stroke_width=STROKE_WIDTH_MM,
        fill="none",
    ))

    # Forward arrow at the last point.
    if el.direction in ("forward", "bidirectional"):
        tip = mm_points[-1]
        prev = mm_points[-2]
        _draw_arrowhead(dwg, tip[0], tip[1], prev[0], prev[1])

    # Backward arrow at the first point.
    if el.direction in ("backward", "bidirectional"):
        tip = mm_points[0]
        prev = mm_points[1]
        _draw_arrowhead(dwg, tip[0], tip[1], prev[0], prev[1])


def assemble(
    flowchart: Flowchart,
    layout: LayoutResult,
) -> tuple[svgwrite.Drawing, list[BrailleLabel], dict]:
    """Assemble the structural SVG, braille labels, and sidecar metadata.

    Returns:
        (svg_drawing, braille_labels, sidecar_dict)
    """
    # Build legend labels from node order.
    label_pairs = [(n.id, n.text or n.id) for n in flowchart.nodes]
    braille_labels = transcribe_labels(label_pairs)

    # Also transcribe connection labels.
    conn_label_pairs = [
        (c.id, c.label) for c in flowchart.connections if c.label
    ]
    if conn_label_pairs:
        conn_braille = transcribe_labels(
            [(cid, lbl) for cid, lbl in conn_label_pairs]
        )
        # Continue numbering from where node labels left off.
        offset = len(braille_labels)
        for bl in conn_braille:
            bl.legend_number = offset + conn_braille.index(bl) + 1
        braille_labels.extend(conn_braille)

    # SVG canvas: graph dimensions in mm plus margin.
    margin_mm = 10.0
    canvas_w = _inches_to_mm(layout.graph_width) + 2 * margin_mm
    canvas_h = _inches_to_mm(layout.graph_height) + 2 * margin_mm

    dwg = svgwrite.Drawing(
        size=(f"{canvas_w}mm", f"{canvas_h}mm"),
        viewBox=f"{-margin_mm} {-margin_mm} {canvas_w} {canvas_h}",
    )

    # Build node_id -> legend_number lookup.
    legend_lookup = {
        label_pairs[i][0]: braille_labels[i].legend_number
        for i in range(len(label_pairs))
    }

    # Draw edges first (so nodes are on top).
    for edge in layout.edges:
        _draw_edge(dwg, edge)

    # Draw nodes.
    for node_id, nl in layout.nodes.items():
        legend_num = legend_lookup.get(node_id, 0)
        _draw_node_shape(dwg, nl, legend_num)

    # Build sidecar metadata.
    sidecar = {
        "source_ir_id": flowchart.id,
        "label_strategy": "legend",
        "bana_conformance": {
            "stroke_width_mm": STROKE_WIDTH_MM,
            "arrow_size_mm": ARROW_SIZE_MM,
            "label_font_mm": LABEL_FONT_MM,
        },
        "legend": [
            {
                "number": bl.legend_number,
                "print_text": bl.print_text,
                "braille_text": bl.braille_text,
            }
            for bl in braille_labels
        ],
    }

    return dwg, braille_labels, sidecar
