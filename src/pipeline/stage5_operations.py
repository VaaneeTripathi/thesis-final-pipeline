"""Stage 5: Operation IR assembly (triggered by Mealy P output).

Maps a VLMOperation + TemporalSegment onto a dict conforming to
operation-schema.json.

The VLM provides classification, semantics and pedagogical context;
the CV TemporalSegment provides sub-second timestamp precision for the
before/after evidence window.
"""
from __future__ import annotations
import datetime
import logging
import math
from typing import Any

from pipeline import config
from pipeline.models import TemporalSegment, VLMOperation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tool_used normalisation (operation-schema enum)
# ---------------------------------------------------------------------------
_TOOL_ENUM = {"marker", "digital pen", "eraser", "pointer", "slide transition", "other"}

def _normalise_tool(raw: str | None) -> str:
    if not raw:
        return "other"
    lower = raw.lower()
    if "eraser" in lower:
        return "eraser"
    if "digital" in lower or "stylus" in lower:
        return "digital pen"
    if "pointer" in lower:
        return "pointer"
    if "slide" in lower:
        return "slide transition"
    if "marker" in lower or "pen" in lower or "chalk" in lower:
        return "marker"
    return "other"


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _seconds_to_mmss(seconds: float) -> str:
    """Convert a float seconds value to MM:SS.mmm format."""
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    m = total_s // 60
    s = total_s % 60
    if ms == 0:
        return f"{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _parse_mmss(ts: str) -> float:
    """Parse MM:SS or MM:SS.mmm to float seconds."""
    try:
        parts = ts.split(":")
        minutes = int(parts[0])
        sec_parts = parts[1].split(".")
        seconds = int(sec_parts[0])
        ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return minutes * 60 + seconds + ms / 1000
    except Exception:
        return 0.0


def _blend_timestamp(vlm_ts: str, cv_ts: float, weight: float = 0.5) -> str:
    """Blend VLM timestamp with CV timestamp for improved precision.

    VLM timestamps are MM:SS from the full-video analysis.
    CV timestamps are precise to the frame. We prefer CV when they are close.
    """
    vlm_s = _parse_mmss(vlm_ts)
    delta = abs(vlm_s - cv_ts)
    if delta < 5.0:
        # CV and VLM agree within 5s — use CV precision
        return _seconds_to_mmss(cv_ts)
    # They diverge significantly — trust VLM (CV temporal segmentation may have missed a boundary)
    log.debug("VLM/CV timestamp divergence: VLM=%.1fs CV=%.1fs", vlm_s, cv_ts)
    return vlm_ts


# ---------------------------------------------------------------------------
# diagram_elements builder
# ---------------------------------------------------------------------------

def _build_diagram_elements(op: VLMOperation) -> list[str]:
    """Produce a flat list of human-readable element descriptions."""
    items: list[str] = []
    for mark_id, desc in op.per_mark_descriptions.items():
        text = desc.get("text") or ""
        role = desc.get("semantic_role") or desc.get("element_type") or ""
        if text:
            items.append(f"Mark [{mark_id}] ({role}): '{text}'")
        else:
            items.append(f"Mark [{mark_id}] ({role})")
    for conn in op.connections:
        fm = conn.get("from_mark")
        tm = conn.get("to_mark")
        label = conn.get("label") or ""
        direction = conn.get("direction", "forward")
        desc = f"Connection from [{fm}] to [{tm}]"
        if label:
            desc += f" labelled '{label}'"
        if direction != "forward":
            desc += f" ({direction})"
        items.append(desc)
    return items


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    vlm_op: VLMOperation,
    segment: TemporalSegment | None,
    video_name: str,
    video_duration: float,
    total_operations: int,
    all_operations: list[VLMOperation],
    time_taken: str = "00:00",
    analysis_timestamp: str | None = None,
) -> dict:
    """Build an operation-schema.json-conformant dict.

    Args:
        vlm_op:            The VLMOperation being assembled.
        segment:           Corresponding CV TemporalSegment (may be None if
                           CV did not find a matching temporal window).
        video_name:        Source video filename.
        video_duration:    Full video duration in seconds (for metadata).
        total_operations:  Total number of operations in the full analysis.
        all_operations:    Full list of VLMOperations (for summary counts).
        time_taken:        Wall-clock analysis time in MM:SS.
        analysis_timestamp: ISO 8601 timestamp string; defaults to utcnow.
    """
    if analysis_timestamp is None:
        analysis_timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # --- Blend CV precision into timestamps ---
    if segment is not None:
        ts_start = _blend_timestamp(vlm_op.timestamp_start, segment.timestamp_start)
        ts_end = _blend_timestamp(vlm_op.timestamp_end, segment.timestamp_end)
    else:
        ts_start = vlm_op.timestamp_start
        ts_end = vlm_op.timestamp_end

    # --- physical_action ---
    raw_action = vlm_op.physical_action or {}
    physical_action = {
        "description": raw_action.get("description", "Instructor modified the whiteboard."),
        "tool_used": _normalise_tool(raw_action.get("tool_used")),
    }

    # --- classification_reasoning ---
    raw_cr = vlm_op.classification_reasoning or {}
    classification_reasoning = {
        "criteria_met": raw_cr.get("criteria_met", []),
        "distinguishing_features": raw_cr.get("distinguishing_features", []),
        "edge_cases_considered": raw_cr.get("edge_cases_considered", []),
    }

    # --- content_description ---
    diagram_elements = _build_diagram_elements(vlm_op)
    content_description = {
        "semantic_content": _build_semantic_content(vlm_op),
        "diagram_elements": diagram_elements,
        "pedagogical_context": vlm_op.pedagogical_context or "",
    }

    # --- visual_evidence ---
    raw_ve = vlm_op.visual_evidence or {}
    visual_evidence: dict[str, Any] = {
        "before_state": raw_ve.get("before_state", ""),
        "after_state": raw_ve.get("after_state", ""),
    }
    if segment is not None:
        visual_evidence["frame_references"] = {
            "start": _seconds_to_mmss(segment.timestamp_start),
            "end": _seconds_to_mmss(segment.timestamp_end),
        }

    # --- summary counts ---
    type_counts = {
        "CREATION": 0,
        "ADDITION": 0,
        "HIGHLIGHTING": 0,
        "ERASURE": 0,
        "COMPLETE_ERASURE": 0,
    }
    for op in all_operations:
        if op.operation_type in type_counts:
            type_counts[op.operation_type] += 1

    duration_str = _seconds_to_mmss(video_duration)

    return {
        "provenance": {
            "model": config.VLM_MODEL,
            "confidence": vlm_op.confidence,
            "visibility_issues": None,
            "time_taken": time_taken,
            "video_file": video_name,
            "frame_rate": 16,
            "analysis_timestamp": analysis_timestamp,
        },
        "analysis": {
            "metadata": {
                "video_duration": duration_str,
                "total_operations_detected": total_operations,
            },
            "operations": [
                {
                    "operation_id": vlm_op.operation_id,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "operation_type": vlm_op.operation_type,
                    "confidence": vlm_op.confidence,
                    "physical_action": physical_action,
                    "classification_reasoning": classification_reasoning,
                    "content_description": content_description,
                    "visual_evidence": visual_evidence,
                }
            ],
            "summary": {
                "creation_count": type_counts["CREATION"],
                "addition_count": type_counts["ADDITION"],
                "highlighting_count": type_counts["HIGHLIGHTING"],
                "erasure_count": type_counts["ERASURE"],
                "complete_erasure_count": type_counts["COMPLETE_ERASURE"],
                "key_observations": [],
                "challenges_encountered": [],
            },
        },
    }


def _build_semantic_content(op: VLMOperation) -> str:
    """Produce a concise semantic summary of the operation."""
    if op.pedagogical_context:
        return op.pedagogical_context

    marks = op.marks_involved
    if not marks:
        return f"{op.operation_type.title()} operation with no identified marks."

    mark_strs = [f"[{m}]" for m in marks]
    texts = [
        op.per_mark_descriptions[m].get("text")
        for m in marks
        if m in op.per_mark_descriptions and op.per_mark_descriptions[m].get("text")
    ]

    base = f"{op.operation_type.title()} involving mark(s) {', '.join(mark_strs)}"
    if texts:
        base += f": {'; '.join(texts)}"
    return base
