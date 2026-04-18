"""Stage 8: Human-readable transcript generation from the operation document.

Produces two output files in output_dir/:

  transcript.txt  — one line per operation:
      [00:38 - 01:35] CREATION: Instructor draws a central "Computer" node...

  transcript.json — structured version with per-entry fields:
      timestamp_start, timestamp_end, operation_type, semantic_content,
      pedagogical_context, diagram_elements
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _format_line(entry: dict) -> str:
    """Format one operation entry as a single transcript line."""
    ts_start = entry.get("timestamp_start", "??:??")
    ts_end = entry.get("timestamp_end", "??:??")
    op_type = entry.get("operation_type", "UNKNOWN")

    content_desc = entry.get("content_description", {})
    semantic = content_desc.get("semantic_content", "")
    pedagogical = content_desc.get("pedagogical_context", "")

    # Prefer pedagogical context for the human-readable summary; fall back to semantic content
    summary = pedagogical or semantic or "(no description)"

    return f"[{ts_start} - {ts_end}] {op_type}: {summary}"


def _build_structured_entry(entry: dict) -> dict:
    """Extract the fields relevant to downstream consumers."""
    content_desc = entry.get("content_description", {})
    return {
        "timestamp_start": entry.get("timestamp_start"),
        "timestamp_end": entry.get("timestamp_end"),
        "operation_type": entry.get("operation_type"),
        "confidence": entry.get("confidence"),
        "semantic_content": content_desc.get("semantic_content", ""),
        "pedagogical_context": content_desc.get("pedagogical_context", ""),
        "diagram_elements": content_desc.get("diagram_elements", []),
    }


def generate(operation_doc: dict, output_dir: Path) -> tuple[Path, Path]:
    """Generate transcript.txt and transcript.json from the operation document.

    Args:
        operation_doc: Single aggregated operation IR dict (from stage5).
        output_dir:    Directory to write transcript files into.

    Returns:
        (txt_path, json_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    operations = operation_doc.get("analysis", {}).get("operations", [])
    if not operations:
        log.warning("Stage 8: operation document contains no operations — empty transcript")

    # --- Plain text transcript ---
    txt_lines: list[str] = []
    provenance = operation_doc.get("provenance", {})
    video_file = provenance.get("video_file", "unknown")
    duration = operation_doc.get("analysis", {}).get("metadata", {}).get("video_duration", "??:??")
    txt_lines.append(f"Lecture Transcript — {video_file}  (duration: {duration})")
    txt_lines.append("=" * 72)
    txt_lines.append("")

    for entry in operations:
        txt_lines.append(_format_line(entry))

    txt_lines.append("")
    summary = operation_doc.get("analysis", {}).get("summary", {})
    key_obs = summary.get("key_observations", [])
    if key_obs:
        txt_lines.append("Key observations:")
        for obs in key_obs:
            txt_lines.append(f"  • {obs}")

    txt_path = output_dir / "transcript.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    log.info("Stage 8: wrote %s (%d operations)", txt_path, len(operations))

    # Structured JSON transcript
    structured = {
        "video_file": video_file,
        "video_duration": duration,
        "operation_count": len(operations),
        "entries": [_build_structured_entry(e) for e in operations],
        "key_observations": key_obs,
    }

    json_path = output_dir / "transcript.json"
    json_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Stage 8: wrote %s", json_path)

    return txt_path, json_path
