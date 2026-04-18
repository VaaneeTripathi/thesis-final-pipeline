"""Stage 5: Operation IR assembly (triggered by Mealy P output).

Two-step API:
  1. build_entry(vlm_op, segment) -> dict
        Produces a single operation object (the inner entry that goes inside
        analysis.operations[]). Called once per Mealy P emission.

  2. assemble_document(entries, all_operations, ...) -> dict
        Wraps all collected entries into one operation-schema.json-conformant
        document for the whole video. Called once at the end of the pipeline.

The VLM provides classification, semantics, and pedagogical context.
The CV TemporalSegment provides sub-second timestamp precision for the
before/after evidence window.
"""
from __future__ import annotations
import datetime
import logging
from typing import Any

from pipeline import config
from pipeline.models import TemporalSegment, VLMOperation

log = logging.getLogger(__name__)


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


def _seconds_to_mmss(seconds: float) -> str:
    """Float seconds → MM:SS.mmm (sub-second precision where needed)."""
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    m = total_s // 60
    s = total_s % 60
    if ms == 0:
        return f"{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _seconds_to_mmss_whole(seconds: float) -> str:
    """Float seconds → MM:SS with whole seconds only.

    The operation schema's video_duration field requires ^\\d{2}:\\d{2}$
    (no milliseconds). Use this function for that field only.
    """
    total_s = int(seconds)
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


def _parse_mmss(ts: str) -> float:
    """Parse MM:SS or MM:SS.mmm → float seconds."""
    try:
        parts = ts.split(":")
        minutes = int(parts[0])
        sec_parts = parts[1].split(".")
        seconds = int(sec_parts[0])
        ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return minutes * 60 + seconds + ms / 1000
    except Exception:
        return 0.0


def _blend_timestamp(vlm_ts: str, cv_ts: float) -> str:
    """Prefer CV precision when VLM and CV timestamps agree within 5 s."""
    vlm_s = _parse_mmss(vlm_ts)
    if abs(vlm_s - cv_ts) < 5.0:
        return _seconds_to_mmss(cv_ts)
    log.debug("VLM/CV timestamp divergence: VLM=%.1fs CV=%.1fs", vlm_s, cv_ts)
    return vlm_ts


def _build_diagram_elements(op: VLMOperation) -> list[str]:
    items: list[str] = []
    for mark_id, desc in op.per_mark_descriptions.items():
        text = desc.get("text") or ""
        role = desc.get("semantic_role") or desc.get("element_type") or ""
        entry = f"Mark [{mark_id}] ({role})"
        if text:
            entry += f": '{text}'"
        items.append(entry)
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


def _build_semantic_content(op: VLMOperation) -> str:
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


def _worst_confidence(operations: list[VLMOperation]) -> str:
    rank = {"high": 0, "medium": 1, "low": 2}
    if not operations:
        return "high"
    return max(operations, key=lambda o: rank.get(o.confidence, 1)).confidence



def build_entry(
    vlm_op: VLMOperation,
    segment: TemporalSegment | None,
) -> dict:
    """Build one operation entry for analysis.operations[].

    This is NOT a full schema document. Collect these entries and pass them
    to assemble_document() once the full video has been processed.
    """
    if segment is not None:
        ts_start = _blend_timestamp(vlm_op.timestamp_start, segment.timestamp_start)
        ts_end = _blend_timestamp(vlm_op.timestamp_end, segment.timestamp_end)
    else:
        ts_start = vlm_op.timestamp_start
        ts_end = vlm_op.timestamp_end

    raw_action = vlm_op.physical_action or {}
    physical_action = {
        "description": raw_action.get("description", "Instructor modified the whiteboard."),
        "tool_used": _normalise_tool(raw_action.get("tool_used")),
    }

    raw_cr = vlm_op.classification_reasoning or {}
    classification_reasoning = {
        "criteria_met": raw_cr.get("criteria_met", []),
        "distinguishing_features": raw_cr.get("distinguishing_features", []),
        "edge_cases_considered": raw_cr.get("edge_cases_considered", []),
    }

    content_description = {
        "semantic_content": _build_semantic_content(vlm_op),
        "diagram_elements": _build_diagram_elements(vlm_op),
        "pedagogical_context": vlm_op.pedagogical_context or "",
    }

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

    return {
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


def assemble_document(
    entries: list[dict],
    all_operations: list[VLMOperation],
    video_name: str,
    video_duration: float,
    fps: float,
    time_taken: str = "00:00",
    analysis_timestamp: str | None = None,
) -> dict:
    """Wrap collected entries into one operation-schema.json-conformant document.

    Args:
        entries:            All dicts from build_entry() calls, in order.
        all_operations:     Full VLMOperation list (for summary counts).
        video_name:         Source video filename (for provenance).
        video_duration:     Full video duration in seconds.
        fps:                Actual video frame rate from VideoIngest.
        time_taken:         Wall-clock analysis time in MM:SS.
        analysis_timestamp: ISO 8601 string; defaults to utcnow.
    """
    if analysis_timestamp is None:
        analysis_timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    
    duration_str = _seconds_to_mmss_whole(video_duration)

    type_counts: dict[str, int] = {
        "CREATION": 0,
        "ADDITION": 0,
        "HIGHLIGHTING": 0,
        "ERASURE": 0,
        "COMPLETE_ERASURE": 0,
    }
    for op in all_operations:
        if op.operation_type in type_counts:
            type_counts[op.operation_type] += 1

    return {
        "provenance": {
            "model": config.VLM_MODEL,
            "confidence": _worst_confidence(all_operations),
            "visibility_issues": None,
            "time_taken": time_taken,
            "video_file": video_name,
            "frame_rate": round(fps),
            "analysis_timestamp": analysis_timestamp,
        },
        "analysis": {
            "metadata": {
                "video_duration": duration_str,
                "total_operations_detected": len(entries),
            },
            "operations": entries,
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
