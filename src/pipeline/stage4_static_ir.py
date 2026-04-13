"""Stage 4: Static IR generation (triggered by Mealy S output).

Merges CV geometry (ElementRegistry — positions, shapes, bboxes) with
VLM semantics (VLMOperation list — text, connections, roles) to produce
a dict conforming to static-schema.json.

The registry contains the board state at pen-lift; operations contains
all VLM operations up to and including the current one so we can build
the latest known description for every mark.
"""
from __future__ import annotations
import datetime
import logging
from typing import Any

from pipeline.models import ElementRegistry, VLMOperation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape mapping: CV shape_type -> static-schema.json enum
# ---------------------------------------------------------------------------
_SHAPE_MAP: dict[str, str] = {
    "rectangle": "rectangle",
    "diamond": "diamond",
    "oval": "oval",
    "circle": "circle",
    "triangle": "triangle",
    "other": "other",
    "connection": "other",   # shouldn't appear as node, but guard it
}

# ---------------------------------------------------------------------------
# Tool-used normalisation for operation-schema enum
# ---------------------------------------------------------------------------
_TOOL_MAP: dict[str, str] = {
    "marker": "marker",
    "pen": "marker",
    "black marker": "marker",
    "digital pen": "digital pen",
    "eraser": "eraser",
    "pointer": "pointer",
    "slide": "slide transition",
    "slide transition": "slide transition",
}

_CONNECTION_SHAPES = {"arrow", "line", "connection"}


def _normalise_tool(raw: str | None) -> str:
    if not raw:
        return "other"
    lower = raw.lower()
    for key, val in _TOOL_MAP.items():
        if key in lower:
            return val
    return "other"


# ---------------------------------------------------------------------------
# Mark info aggregation
# ---------------------------------------------------------------------------

def _build_mark_info(
    operations: list[VLMOperation],
) -> dict[int, dict[str, Any]]:
    """Return the latest VLM description for each mark ID.

    Scans operations in order so later operations overwrite earlier ones
    for the same mark — giving us the most up-to-date description.
    """
    info: dict[int, dict[str, Any]] = {}
    for op in operations:
        for mark_id, desc in op.per_mark_descriptions.items():
            info[mark_id] = desc
    return info


def _collect_connections(operations: list[VLMOperation]) -> list[dict]:
    """Collect all VLM-described connections, deduplicating by (from, to, direction)."""
    seen: set[tuple] = set()
    result: list[dict] = []
    for op in operations:
        for conn in op.connections:
            key = (conn.get("from_mark"), conn.get("to_mark"), conn.get("direction"))
            if key not in seen:
                seen.add(key)
                result.append(conn)
    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    registry: ElementRegistry,
    operations: list[VLMOperation],
    snapshot_timestamp: float,
    video_name: str,
    time_taken: str = "00:00",
) -> dict:
    """Build a static-schema.json-conformant dict from CV + VLM data.

    Args:
        registry:           Active ElementRegistry at pen-lift.
        operations:         All VLMOperations up to and including the current one.
        snapshot_timestamp: Board timestamp (seconds) for this snapshot.
        video_name:         Source video filename (for provenance).
        time_taken:         Wall-clock analysis duration in MM:SS.

    Returns:
        dict conforming to static-schema.json.
    """
    mark_info = _build_mark_info(operations)
    vlm_connections = _collect_connections(operations)

    # Confidence: lowest confidence among contributing operations, or "high" if none
    if operations:
        confidence_rank = {"high": 0, "medium": 1, "low": 2}
        worst = max(operations, key=lambda o: confidence_rank.get(o.confidence, 1))
        overall_confidence = worst.confidence
    else:
        overall_confidence = "high"

    # --- elements.nodes ---
    nodes: list[dict] = []
    for mark_id, region in sorted(registry.elements.items()):
        if region.shape_type in _CONNECTION_SHAPES:
            continue  # handled as connections below

        info = mark_info.get(mark_id, {})
        element_type = info.get("element_type", "node")
        if element_type == "connection":
            continue  # VLM says this is a connection; skip from nodes

        x, y, w, h = region.bbox
        shape = _SHAPE_MAP.get(region.shape_type, "other")

        node: dict = {
            "id": f"mark-{mark_id}",
            "shape": shape,
            "text": info.get("text"),
            "position": {"x": float(x + w // 2), "y": float(y + h // 2)},
        }
        nodes.append(node)

    # --- elements.connections ---
    connections: list[dict] = []
    conn_idx = 0
    for conn in vlm_connections:
        from_id = conn.get("from_mark")
        to_id = conn.get("to_mark")
        if from_id is None or to_id is None:
            continue

        direction = conn.get("direction", "forward")
        if direction not in {"forward", "backward", "bidirectional", "none"}:
            direction = "forward"

        line_type_raw = conn.get("line_type", "solid")
        line_type = "solid" if "solid" in line_type_raw else (
            "dashed" if "dash" in line_type_raw else (
                "dotted" if "dot" in line_type_raw else "solid"
            )
        )

        entry: dict = {
            "id": f"conn-{conn_idx}",
            "source": f"mark-{from_id}",
            "target": f"mark-{to_id}",
            "direction": direction,
            "line_type": line_type,
        }
        if conn.get("label"):
            entry["label"] = conn["label"]
        connections.append(entry)
        conn_idx += 1

    # --- semantics.symbols ---
    shape_meanings: dict[str, str] = {
        "rectangle": "process / action step",
        "diamond": "decision point",
        "oval": "start or end terminal",
        "circle": "connector / junction",
        "triangle": "off-page connector",
        "other": "custom element",
    }
    used_shapes = {_SHAPE_MAP.get(r.shape_type, "other") for r in registry.elements.values()}
    symbols = [
        {
            "symbol": shape,
            "meaning": meaning,
            "applicable_to": [
                f"mark-{mid}" for mid, reg in registry.elements.items()
                if _SHAPE_MAP.get(reg.shape_type, "other") == shape
            ],
        }
        for shape, meaning in shape_meanings.items()
        if shape in used_shapes
    ]

    # --- semantics.annotations (from HIGHLIGHTING operations) ---
    annotations: list[dict] = []
    ann_idx = 0
    for op in operations:
        if op.operation_type == "HIGHLIGHTING":
            for mark_id in op.marks_involved:
                annotations.append({
                    "id": f"ann-{ann_idx}",
                    "target": f"mark-{mark_id}",
                    "annotation_type": "highlight",
                    "content": op.pedagogical_context or None,
                })
                ann_idx += 1

    # --- state.board_state ---
    if operations:
        last_op = operations[-1]
        board_state = last_op.pedagogical_context or (
            f"Board at {snapshot_timestamp:.1f}s: "
            f"{len(registry.elements)} elements visible."
        )
    else:
        board_state = f"Board at {snapshot_timestamp:.1f}s: {len(registry.elements)} elements visible."

    return {
        "id": datetime.datetime.utcfromtimestamp(snapshot_timestamp).isoformat() + "Z",
        "elements": {
            "nodes": nodes,
            "connections": connections,
        },
        "structure": {
            "hierarchy": [],
        },
        "state": {
            "board_state": board_state,
        },
        "semantics": {
            "symbols": symbols if symbols else None,
            "annotations": annotations if annotations else None,
        },
        "provenance": {
            "model": "gemini-2.5-flash + opencv-cv",
            "confidence": overall_confidence,
            "visibility_issues": None,
            "time_taken": time_taken,
        },
    }
