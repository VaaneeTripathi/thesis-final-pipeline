"""Stage 2: VLM analysis on the SoM-marked video.

Uploads the full marked video to Gemini 2.5 Flash with:
  - The element registry as SoM context (Mark [N] = shape at (x, y))
  - The operation taxonomy, decision tree, and few-shot example
  - Structured JSON output spec referencing mark IDs

After the initial analysis a verification pass re-sends the video
alongside the generated output and asks the model to confirm or correct
it (ported from flowchart-description-genai/flowchart_analyser.py).

Response is cached to output_dir/vlm_cache.json to avoid re-running
the API call during development.
"""
from __future__ import annotations
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import google.generativeai as genai

from pipeline import config
from pipeline.models import ElementRegistry, VLMOperation

log = logging.getLogger(__name__)

_CACHE_FILENAME = "vlm_cache.json"


# ---------------------------------------------------------------------------
# Registry serialisation for the prompt
# ---------------------------------------------------------------------------

def _registry_to_prompt_lines(registry: ElementRegistry) -> str:
    """Describe every active mark to the VLM in a compact numbered list."""
    if not registry.elements:
        return "No elements were detected by the CV stage."

    lines: list[str] = [
        "The following visual elements were detected and marked on every frame",
        "of the video by the computer-vision pre-processing stage.",
        "Each element has a numbered label overlaid directly on the video.",
        "",
    ]
    for mark_id, region in sorted(registry.elements.items()):
        x, y, w, h = region.bbox
        cx, cy = region.centroid
        lines.append(
            f"  Mark [{mark_id}]: shape={region.shape_type}, "
            f"bbox=(x={x}, y={y}, w={w}, h={h}), "
            f"centroid=({cx}, {cy}), "
            f"first_seen={region.first_seen:.1f}s"
        )
    lines.append(
        "\nWhen referencing elements, ALWAYS use the mark number, e.g. 'Mark [3] was added'."
    )
    lines.append(
        "If you see connections between elements not matched to any mark, "
        "describe them as: 'unmarked connection from [N] to [M]'."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt components
# ---------------------------------------------------------------------------

_TAXONOMY = """\
OPERATION TAXONOMY
Classify each detected change into exactly ONE of these five types:

1. CREATION  — initial drawing on blank/erased surface
   ALL must be true: preceded by blank slate OR complete erasure; new flowchart
   elements appear; no pre-existing components remain visible.

2. ADDITION  — new nodes, arrows, or text added to an existing diagram
   ALL must be true: pre-existing flowchart visible; NO erasure since last op;
   new content appears; new elements provide NEW semantic content (not emphasis).

3. HIGHLIGHTING — emphasis/annotation overlaid on existing elements
   ALL must be true: pre-existing elements unchanged; marks OVERLAID on existing
   content; purpose is emphasis/enumeration/reference, NOT new content.

4. ERASURE   — specific elements removed; overall diagram survives
   ALL must be true: pre-existing flowchart visible; specific elements removed;
   other parts remain intact.

5. COMPLETE_ERASURE — entire diagram wiped; results in blank surface
   ALL must be true: previously visible flowchart; ALL components removed;
   results in blank surface; marks end of diagrammatic context.

DECISION TREE
Start: change detected
├─ Any pre-existing flowchart? NO → CREATION
└─ YES
   ├─ Something removed? YES
   │  ├─ Entire diagram gone? YES → COMPLETE_ERASURE
   │  └─ NO → ERASURE
   └─ Something added?
      ├─ New semantic content? YES → ADDITION
      └─ NO → HIGHLIGHTING
"""

_SCHEMA = """\
OUTPUT SCHEMA
Respond with ONLY valid JSON — no preamble, no markdown fences.

{
  "metadata": {
    "video_duration": "MM:SS",
    "total_operations_detected": <int>,
    "analysis_confidence": "high|medium|low",
    "visibility_issues": "<string>"
  },
  "operations": [
    {
      "operation_id": <int, 1-based>,
      "timestamp_start": "MM:SS",
      "timestamp_end": "MM:SS",
      "operation_type": "CREATION|ADDITION|HIGHLIGHTING|ERASURE|COMPLETE_ERASURE",
      "confidence": "high|medium|low",
      "marks_involved": [<mark_id>, ...],
      "per_mark_descriptions": {
        "<mark_id>": {
          "text": "<text content visible inside/near this mark, or null>",
          "element_type": "node|connection|annotation",
          "semantic_role": "<what this element represents in the diagram>"
        }
      },
      "connections": [
        {
          "from_mark": <mark_id>,
          "to_mark": <mark_id>,
          "direction": "forward|backward|bidirectional",
          "label": "<arrow label or null>",
          "line_type": "arrow|line"
        }
      ],
      "classification_reasoning": {
        "criteria_met": ["<criterion>"],
        "distinguishing_features": ["<feature>"],
        "edge_cases_considered": ["<edge case>"]
      },
      "pedagogical_context": "<why this operation occurred in the lecture>",
      "physical_action": {
        "description": "<what the instructor physically did>",
        "tool_used": "hand|eraser|marker|pointer|other"
      },
      "visual_evidence": {
        "before_state": "<board state before operation>",
        "after_state": "<board state after operation>",
        "frame_references": "<timestamp or frame range for verification>"
      }
    }
  ],
  "summary": {
    "creation_count": <int>,
    "addition_count": <int>,
    "highlighting_count": <int>,
    "erasure_count": <int>,
    "complete_erasure_count": <int>,
    "key_observations": ["<observation>"],
    "challenges_encountered": ["<challenge>"]
  }
}
"""

# Few-shot example ported from llm-action-detection-api/agentic-analyser.py
_FEW_SHOT_EXAMPLE = """\
FEW-SHOT EXAMPLE (reference format only — do not copy timestamps or content):

{
  "metadata": {
    "video_duration": "03:45",
    "total_operations_detected": 3,
    "analysis_confidence": "high",
    "visibility_issues": "None - clear board visibility throughout"
  },
  "operations": [
    {
      "operation_id": 1,
      "timestamp_start": "00:15",
      "timestamp_end": "00:47",
      "operation_type": "CREATION",
      "confidence": "high",
      "marks_involved": [1, 2, 3],
      "per_mark_descriptions": {
        "1": {"text": "Client", "element_type": "node", "semantic_role": "initiating component"},
        "2": {"text": "Server", "element_type": "node", "semantic_role": "responding component"},
        "3": {"text": "HTTP Request/Response", "element_type": "connection", "semantic_role": "communication channel"}
      },
      "connections": [
        {"from_mark": 1, "to_mark": 2, "direction": "bidirectional", "label": "HTTP Request/Response", "line_type": "arrow"}
      ],
      "classification_reasoning": {
        "criteria_met": [
          "Preceded by blank slate",
          "New flowchart elements appear forming a coherent diagram",
          "No pre-existing flowchart components remain visible"
        ],
        "distinguishing_features": [
          "Board completely empty at start",
          "First strokes establish diagram structure"
        ],
        "edge_cases_considered": []
      },
      "pedagogical_context": "Introducing basic client-server architecture at lecture start",
      "physical_action": {
        "description": "Instructor draws two rectangles and a bidirectional arrow using black marker",
        "tool_used": "marker"
      },
      "visual_evidence": {
        "before_state": "Completely blank whiteboard",
        "after_state": "Three-element diagram: Mark [1] Client, Mark [2] Server, Mark [3] arrow between them",
        "frame_references": "Start: 00:15, nodes complete: 00:32, arrow complete: 00:47"
      }
    }
  ],
  "summary": {
    "creation_count": 1,
    "addition_count": 0,
    "highlighting_count": 0,
    "erasure_count": 0,
    "complete_erasure_count": 0,
    "key_observations": ["Single creation operation establishing lecture foundation"],
    "challenges_encountered": []
  }
}
"""

_ANALYSIS_PROTOCOL = """\
ANALYSIS PROTOCOL
1. Watch the full video once to understand the lecture arc and count total operations.
2. For each operation:
   a. Identify exact start and end timestamps (first visible change → completion).
   b. Apply the decision tree to classify operation type.
   c. List ALL mark IDs involved in marks_involved.
   d. For every involved mark, fill per_mark_descriptions.
   e. Describe all connections between numbered marks.
3. Validate: summary counts must match the operations array length by type.
4. Output ONLY the JSON — no preamble, no explanation outside the JSON.

CRITICAL REQUIREMENTS
- ONLY report operations with clear visual evidence in the video.
- DO NOT hallucinate operations not visible in the video.
- Reference elements by Mark [N] number throughout.
- Timestamps: MM:SS format (subsecond precision MM:SS.mmm if possible).
- If an element is visible but has no mark number, describe it as 'unmarked'.
"""


def _build_analysis_prompt(registry: ElementRegistry) -> str:
    registry_section = _registry_to_prompt_lines(registry)
    return (
        "TASK: Flowchart Operation Detection on SoM-Marked Lecture Video\n\n"
        "CONTEXT: SET-OF-MARKS PRE-PROCESSING\n"
        "Every frame of this video has been pre-processed by a computer-vision\n"
        "pipeline that detected whiteboard elements and overlaid numbered\n"
        "bounding-box labels (Set-of-Marks). Reference elements by mark number.\n\n"
        "DETECTED MARKS\n"
        f"{registry_section}\n\n"
        f"{_TAXONOMY}\n\n"
        f"{_ANALYSIS_PROTOCOL}\n\n"
        f"{_FEW_SHOT_EXAMPLE}\n\n"
        f"{_SCHEMA}\n"
    )


def _build_verification_prompt(analysis_json: str) -> str:
    """Verification pass prompt (ported from flowchart_analyser.py verify_diagram)."""
    return (
        "TASK: Verification of Flowchart Operation Analysis\n\n"
        "You previously analysed this SoM-marked lecture video and produced the\n"
        "following operation log. Watch the video again and verify it.\n\n"
        "GENERATED ANALYSIS:\n"
        f"{analysis_json}\n\n"
        "VERIFICATION CHECKLIST\n"
        "1. Are all operations listed? Are any missing?\n"
        "2. Are the timestamps accurate (first visible change → completion)?\n"
        "3. Are the operation types correctly classified per the decision tree?\n"
        "4. Do marks_involved and per_mark_descriptions reference the correct marks?\n"
        "5. Do summary counts match the operations array?\n\n"
        "OUTPUT\n"
        "If the analysis is correct, return it unchanged.\n"
        "If corrections are needed, return the corrected complete JSON.\n"
        "Output ONLY valid JSON — no preamble, no explanation outside the JSON.\n"
        f"{_SCHEMA}\n"
    )


# ---------------------------------------------------------------------------
# Upload / poll helpers (ported from llm-action-detection-api/agentic-analyser.py)
# ---------------------------------------------------------------------------

def _upload_with_retry(video_path: Path, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            video_file = genai.upload_file(path=str(video_path))
            log.info("Uploaded %s as %s", video_path.name, video_file.name)
            return video_file
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            log.warning(
                "Upload attempt %d failed: %s — retrying in %.1fs",
                attempt + 1, exc, wait,
            )
            time.sleep(wait)


def _wait_for_processing(video_file, timeout: float = 300.0):
    start = time.time()
    while video_file.state.name == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError(f"Video processing timed out after {timeout}s")
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError("Gemini video processing FAILED")
    return video_file


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract JSON from VLM response — handles fenced and bare objects."""
    # Try ```json ... ``` block first (in case model disobeys "no fences" instruction)
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if blocks:
        return json.loads(blocks[-1])

    # Try bare JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No JSON found in VLM response")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_operations(raw: dict) -> list[VLMOperation]:
    ops: list[VLMOperation] = []
    for item in raw.get("operations", []):
        # per_mark_descriptions keys are strings in JSON; convert to int
        pmd_raw = item.get("per_mark_descriptions", {})
        pmd = {int(k): v for k, v in pmd_raw.items()}

        ops.append(VLMOperation(
            operation_id=item["operation_id"],
            operation_type=item["operation_type"],
            timestamp_start=item["timestamp_start"],
            timestamp_end=item["timestamp_end"],
            confidence=item.get("confidence", "medium"),
            marks_involved=[int(m) for m in item.get("marks_involved", [])],
            per_mark_descriptions=pmd,
            connections=item.get("connections", []),
            classification_reasoning=item.get("classification_reasoning", {}),
            pedagogical_context=item.get("pedagogical_context", ""),
            physical_action=item.get("physical_action", {}),
            visual_evidence=item.get("visual_evidence", {}),
        ))
    return ops


def _collect_text(response) -> str:
    text_parts = [p.text for p in response.parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts) or response.text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    marked_video_path: Path,
    registry: ElementRegistry,
    output_dir: Path,
    verify: bool = True,
) -> list[VLMOperation]:
    """Upload the marked video to Gemini and parse the operation log.

    Args:
        marked_video_path: Path to the SoM-marked video from Stage 1.
        registry: ElementRegistry from Stage 1 (mark IDs + regions).
        output_dir: Directory for the VLM response cache.
        verify: If True, run a second verification pass (ported from
                flowchart_analyser.py) before returning.

    Returns:
        list[VLMOperation] parsed from the VLM response.

    The raw response is cached to output_dir/vlm_cache.json — delete this
    file to force a fresh API call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / _CACHE_FILENAME

    if cache_path.exists():
        log.info("Stage 2: loading cached VLM response from %s", cache_path)
        raw = json.loads(cache_path.read_text())
        return _parse_operations(raw)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name=config.VLM_MODEL)

    log.info("Stage 2: uploading %s to Gemini...", marked_video_path.name)
    video_file = _upload_with_retry(marked_video_path)

    try:
        log.info("Stage 2: waiting for Gemini video processing...")
        video_file = _wait_for_processing(video_file)

        # --- Initial analysis pass ---
        analysis_prompt = _build_analysis_prompt(registry)
        log.info("Stage 2: initial analysis (model=%s)...", config.VLM_MODEL)
        chat = model.start_chat()
        response = chat.send_message(
            [video_file, analysis_prompt],
            request_options={"timeout": config.VLM_TIMEOUT},
        )
        response_text = _collect_text(response)
        raw = _extract_json(response_text)
        log.info(
            "Stage 2: initial analysis complete — %d operations",
            len(raw.get("operations", [])),
        )

        # --- Verification pass (ported from flowchart_analyser.py) ---
        if verify:
            log.info("Stage 2: verification pass...")
            verification_prompt = _build_verification_prompt(json.dumps(raw, indent=2))
            v_response = chat.send_message(
                [video_file, verification_prompt],
                request_options={"timeout": config.VLM_TIMEOUT},
            )
            v_text = _collect_text(v_response)
            try:
                verified_raw = _extract_json(v_text)
                raw = verified_raw
                log.info(
                    "Stage 2: verification complete — %d operations after review",
                    len(raw.get("operations", [])),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                log.warning(
                    "Stage 2: verification response was not valid JSON (%s) — keeping initial analysis",
                    exc,
                )

    finally:
        try:
            genai.delete_file(video_file.name)
            log.info("Stage 2: deleted uploaded file %s", video_file.name)
        except Exception as exc:
            log.warning("Stage 2: failed to delete uploaded file: %s", exc)

    cache_path.write_text(json.dumps(raw, indent=2))
    log.info("Stage 2: cached response -> %s", cache_path)

    operations = _parse_operations(raw)
    log.info("Stage 2: parsed %d operations", len(operations))
    return operations
