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
import io
import json
import logging
import os
import random
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from pipeline import config
from pipeline.models import BoardSnapshot, ElementRegistry, KeyframeAnnotation, VLMOperation

log = logging.getLogger(__name__)

_CACHE_FILENAME = "vlm_cache.json"


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


def _keyframe_context(keyframes: list[KeyframeAnnotation]) -> str:
    """Build the TEMPORAL REFERENCE KEYFRAMES section for the analysis prompt.

    Lists each pen-lift keyframe's timestamp and CV-detected marks. The
    corresponding PNG images are passed separately in the message content list,
    giving the VLM both visual and positional grounding at each key moment.
    """
    if not keyframes:
        return ""
    lines = [
        "TEMPORAL REFERENCE KEYFRAMES",
        "Still images of the board at each pen-lift are attached after this video.",
        "They show the exact board state with numbered SoM marks overlaid.",
        "Use them to: verify mark IDs match visual elements, confirm board state",
        "at each pen-lift, and cross-check your operation timestamps.\n",
    ]
    for kf in keyframes:
        lines.append(f"Keyframe {kf.segment_id}  (pen-lift at {kf.timestamp:.1f}s):")
        for mark in sorted(kf.marks, key=lambda m: m["mark_id"]):
            cx, cy = mark["centroid"]
            lines.append(
                f"  Mark [{mark['mark_id']}]: {mark['shape_type']}  centroid=({cx},{cy})"
            )
        lines.append("")
    return "\n".join(lines)


def _build_analysis_prompt(
    registry: ElementRegistry,
    keyframes: list[KeyframeAnnotation] | None = None,
) -> str:
    registry_section = _registry_to_prompt_lines(registry)
    keyframe_section = _keyframe_context(keyframes or [])
    return (
        "TASK: Flowchart Operation Detection on Lecture Video\n\n"
        "CONTEXT: SET-OF-MARKS SPATIAL GROUNDING\n"
        "The computer-vision pipeline detected whiteboard elements and assigned\n"
        "each a numbered mark ID. The original clean video is provided first,\n"
        "followed by still keyframe images taken at each pen-lift moment with\n"
        "SoM marks overlaid. Reference ALL elements by mark number.\n\n"
        "DETECTED MARKS (full registry)\n"
        f"{registry_section}\n\n"
        + (f"{keyframe_section}\n" if keyframe_section else "")
        + f"{_TAXONOMY}\n\n"
        f"{_ANALYSIS_PROTOCOL}\n\n"
        f"{_FEW_SHOT_EXAMPLE}\n\n"
        f"{_SCHEMA}\n"
    )


def _build_verification_prompt(
    analysis_json: str,
    keyframes: list[KeyframeAnnotation] | None = None,
) -> str:
    """Verification pass prompt (ported from flowchart_analyser.py verify_diagram)."""
    keyframe_section = _keyframe_context(keyframes or [])
    return (
        "TASK: Verification of Flowchart Operation Analysis\n\n"
        "You previously analysed this lecture video and produced the following\n"
        "operation log. The original video and keyframe stills are provided again.\n\n"
        + (f"{keyframe_section}\n" if keyframe_section else "")
        + "GENERATED ANALYSIS:\n"
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


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def _no_thinking() -> types.GenerateContentConfig:
    """Disable extended thinking on gemini-2.5-flash.

    Without thinking_budget=0, the model can spend 20+ minutes on internal
    chain-of-thought before emitting any tokens, causing gRPC deadline errors.
    """
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )


def _video_part(video_file) -> types.Part:
    return types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4")


def _image_part(image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def _upload_with_retry(client: genai.Client, video_path: Path, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            video_file = client.files.upload(file=video_path)
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


def _wait_for_processing(client: genai.Client, video_file, timeout: float = 300.0):
    start = time.time()
    while video_file.state.name == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError(f"Video processing timed out after {timeout}s")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError("Gemini video processing FAILED")
    return video_file



def _extract_json(text: str) -> dict:
    """Extract the best valid JSON object from a VLM response.

    Strategy (in order):
      1. All ```json ... ``` fenced blocks — try each from last to first so we
         pick the most complete one; a truncated last block falls back to earlier ones.
      2. Largest bare {...} object in the response.
    Logs a debug excerpt around the failure position to aid diagnosis.
    """
    errors: list[str] = []

    # Fenced blocks — try last → first (last is usually most complete, but may be truncated)
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError as exc:
            pos = exc.pos
            snippet = block[max(0, pos - 80): pos + 80].replace("\n", "↵")
            errors.append(f"fenced block: {exc.msg} at char {pos} — …{snippet}…")

    # Bare JSON object — find the longest {...} span
    for match in sorted(re.finditer(r"\{[\s\S]*\}", text), key=lambda m: len(m.group()), reverse=True):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            pos = exc.pos
            s = match.group(0)
            snippet = s[max(0, pos - 80): pos + 80].replace("\n", "↵")
            errors.append(f"bare object: {exc.msg} at char {pos} — …{snippet}…")
            break  # only try the largest match

    for err in errors:
        log.debug("_extract_json attempt failed: %s", err)
    log.warning(
        "_extract_json: could not parse JSON from response (%d chars). "
        "First 300: %s",
        len(text), text[:300].replace("\n", " "),
    )
    raise ValueError(f"No valid JSON found in VLM response. Attempts: {errors}")



def _coerce_mark_id(m) -> int | None:
    if isinstance(m, (int, float)):
        return int(m)
    if isinstance(m, str):
        try:
            return int(m)
        except ValueError:
            return None
    if isinstance(m, dict):
        for key in ("mark_id", "id", "mark", "number"):
            if key in m:
                try:
                    return int(m[key])
                except (ValueError, TypeError):
                    pass
        # Last resort: first integer value in the dict
        for v in m.values():
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
    return None


def _parse_operations(raw: dict) -> list[VLMOperation]:
    ops: list[VLMOperation] = []
    for item in raw.get("operations", []):
        # per_mark_descriptions keys are strings in JSON; convert to int.
        # Guard against non-integer keys (e.g. "mark_1") the model sometimes returns.
        pmd_raw = item.get("per_mark_descriptions", {})
        pmd: dict = {}
        for k, v in pmd_raw.items():
            try:
                pmd[int(k)] = v
            except (ValueError, TypeError):
                log.debug("_parse_operations: non-integer per_mark_descriptions key %r — skipped", k)

        marks_involved = [
            mid for m in item.get("marks_involved", [])
            if (mid := _coerce_mark_id(m)) is not None
        ]

        ops.append(VLMOperation(
            operation_id=item["operation_id"],
            operation_type=item["operation_type"],
            timestamp_start=item["timestamp_start"],
            timestamp_end=item["timestamp_end"],
            confidence=item.get("confidence", "medium"),
            marks_involved=marks_involved,
            per_mark_descriptions=pmd,
            connections=item.get("connections", []),
            classification_reasoning=item.get("classification_reasoning", {}),
            pedagogical_context=item.get("pedagogical_context", ""),
            physical_action=item.get("physical_action", {}),
            visual_evidence=item.get("visual_evidence", {}),
        ))
    return ops


def _retry_delay_seconds(exc: genai_errors.ClientError) -> int:
    """Extract the retryDelay value from a 429 ClientError response, or return 65."""
    try:
        details = exc.details.get("error", {}).get("details", [])
        for d in details:
            delay = d.get("retryDelay", "")
            if delay:
                return int(str(delay).rstrip("s")) + 5  # +5s buffer
    except Exception:
        pass
    return 65


def _stream_generate(client: genai.Client, contents: list, label: str) -> str:
    """Stream tokens from the model with thinking disabled.
    """
    log.info("Stage 2: %s (streaming)...", label)
    for attempt in range(3):
        try:
            chunks: list[str] = []
            for chunk in client.models.generate_content_stream(
                model=config.VLM_MODEL,
                contents=contents,
                config=_no_thinking(),
            ):
                try:
                    if chunk.text:
                        chunks.append(chunk.text)
                except Exception:
                    pass  # chunks carrying only safety ratings have no .text
            return "".join(chunks)
        except genai_errors.ClientError as exc:
            if exc.code == 429 and attempt < 2:
                wait = _retry_delay_seconds(exc)
                log.warning(
                    "Stage 2: %s rate-limited (429) — waiting %ds then retrying (attempt %d/3)...",
                    label, wait, attempt + 2,
                )
                time.sleep(wait)
            else:
                raise


def run(
    video_path: Path,
    registry: ElementRegistry,
    output_dir: Path,
    keyframes: list[KeyframeAnnotation] | None = None,
    verify: bool = True,
) -> list[VLMOperation]:
    """Upload the original lecture video (+ keyframe stills) to Gemini and parse operations.

    Args:
        video_path:  Path to the original clean video (no SoM overlay).
        registry:    ElementRegistry from Stage 1 (mark IDs + regions).
        output_dir:  Directory for the VLM response cache.
        keyframes:   KeyframeAnnotation list from Stage 1. Each PNG is uploaded
                     alongside the video to give the VLM per-pen-lift grounding.
        verify:      If True, run a second verification pass before returning.

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

    client = _get_client()

    log.info("Stage 2: uploading %s to Gemini...", video_path.name)
    video_file = _upload_with_retry(client, video_path)
    all_uploaded = [video_file]

    try:
        log.info("Stage 2: waiting for Gemini video processing...")
        video_file = _wait_for_processing(client, video_file)

        # Video + text prompt only — the registry description encodes all SoM context.
        # Keyframes are sent one-at-a-time in analyse_snapshots() (stage 2b).
        analysis_prompt = _build_analysis_prompt(registry, keyframes)
        content = [_video_part(video_file), analysis_prompt]

        response_text = _stream_generate(client, content, "initial analysis")
        raw = _extract_json(response_text)
        log.info(
            "Stage 2: initial analysis complete — %d operations",
            len(raw.get("operations", [])),
        )

        # Cache immediately so a crash during verification doesn't lose this result.
        # If verification succeeds the cache is overwritten with the improved version.
        cache_path.write_text(json.dumps(raw, indent=2))

        # Verification pass — prompt embeds initial JSON so no chat state needed.
        if verify:
            verification_prompt = _build_verification_prompt(
                json.dumps(raw, indent=2), keyframes
            )
            v_content = [_video_part(video_file), verification_prompt]
            v_text = _stream_generate(client, v_content, "verification pass")
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
        for uploaded in all_uploaded:
            try:
                client.files.delete(name=uploaded.name)
                log.debug("Stage 2: deleted uploaded file %s", uploaded.name)
            except Exception as exc:
                log.warning("Stage 2: failed to delete %s: %s", uploaded.name, exc)
        log.info("Stage 2: cleaned up %d uploaded file(s)", len(all_uploaded))

    cache_path.write_text(json.dumps(raw, indent=2))
    log.info("Stage 2: cached response -> %s", cache_path)

    operations = _parse_operations(raw)
    log.info("Stage 2: parsed %d operations", len(operations))
    return operations


_SNAPSHOT_SCHEMA = """\
SNAPSHOT OUTPUT SCHEMA
Respond with ONLY valid JSON — no preamble, no markdown fences.

{
  "mark_descriptions": {
    "<mark_id>": {
      "text": "<all text visible inside or adjacent to this mark, or null>",
      "shape": "rectangle|diamond|oval|parallelogram|circle|rounded-rectangle|triangle|other",
      "element_type": "node|connection|annotation",
      "semantic_role": "<what this element represents in the diagram, e.g. 'start terminal', 'decision point', 'process step'>",
      "visual": {"color": "<dominant non-black/white color if present, e.g. 'red', 'blue', else null>"}
    }
  },
  "connections": [
    {
      "from_mark": <int>,
      "to_mark": <int>,
      "direction": "forward|backward|bidirectional|none",
      "line_type": "solid|dashed|dotted",
      "label": "<text on or beside the arrow/line, or null>"
    }
  ],
  "symbol_meanings": [
    {
      "shape": "<shape name matching the enum above>",
      "meaning": "<what this shape convention represents in this diagram>"
    }
  ],
  "groupings": [
    {
      "label": "<descriptive group name>",
      "parent": "<parent group label for nested groups, or null for top-level>",
      "members": [<mark_id>, ...]
    }
  ],
  "annotations": [
    {
      "mark_id": <int>,
      "annotation_type": "container|separator|highlight|circle|underline|arrow|cloud|strikethrough",
      "content": "<text attached to the annotation, or null>"
    }
  ],
  "cross_links": [
    {
      "source_flowchart": "<label or description of the flowchart containing the source>",
      "target_flowchart": "<label or description of the flowchart containing the target>",
      "source_element": "<description of source element in its flowchart>",
      "target_element": "<description of target element in its flowchart>",
      "label": "<relationship label, or null>"
    }
  ],
  "board_state": "<complete natural-language description of the full board>",
  "confidence": "high|medium|low",
  "visibility_issues": "<description of any glare, occlusion, or legibility problems, or null>"
}
"""

_SNAPSHOT_RULES = """\
ANALYSIS RULES

MARK DESCRIPTIONS — required for every mark in the CV list:
  text        : read all text inside or near the mark; null if none visible
  shape       : use the enum — CORRECT the CV shape if you see it more clearly
                (the CV shape shown in the marks list is a hint, not ground truth)
  element_type: node = standalone flowchart element; connection = arrow or line;
                annotation = emphasis mark overlaid on existing content
  semantic_role: describe the element's function in this diagram
  visual.color: note a distinctive non-black/white fill or stroke color; null otherwise

CONNECTIONS — edges between marks:
  Only report connections where BOTH endpoints are numbered marks in the CV list.
  direction : forward = arrowhead points from→to; backward = arrowhead points to→from;
              bidirectional = arrowheads at both ends; none = undirected line
  line_type : solid = unbroken; dashed = evenly broken segments; dotted = dot pattern

SYMBOL MEANINGS — shape legend:
  Only include shapes that actually appear in this snapshot.
  Infer meaning from diagram context (e.g. diamonds at branch points → "decision point").
  Omit entirely if the diagram convention cannot be determined with confidence.

GROUPINGS — logical hierarchy:
  Only include if the board shows clear groupings: swim lanes, explicit group boxes,
  sub-flow regions with labels, or numbered stages.
  Omit if the diagram is flat with no visible grouping structure.

ANNOTATIONS — emphasis overlays:
  Only report marks drawn ON TOP OF existing content for emphasis.
  Do NOT report regular nodes or connections as annotations.

CROSS-LINKS — multiple distinct flowcharts:
  Only include if two or more clearly distinct diagrams share the board with a
  visible connecting relationship between them.
  Omit for a single connected diagram.

Output ONLY valid JSON. Empty lists [] for sections with no items. null for unknown values.
"""


def _build_snapshot_prompt(keyframe: KeyframeAnnotation, registry: ElementRegistry) -> str:
    """Build the per-keyframe static IR prompt."""
    lines = [
        f"TASK: Whiteboard Snapshot Analysis for Accessible IR",
        f"Snapshot timestamp: {keyframe.timestamp:.1f}s\n",
        "CV-DETECTED MARKS",
        "The following elements were detected by the computer-vision pipeline.",
        "Their positions and bounding boxes are ground truth. Shapes are CV estimates",
        "— correct them in mark_descriptions if you see the shape more clearly.\n",
    ]
    for mark in sorted(keyframe.marks, key=lambda m: m["mark_id"]):
        mid = mark["mark_id"]
        region = registry.elements.get(mid)
        if region is None:
            continue
        x, y, w, h = region.bbox
        cx, cy = region.centroid
        lines.append(
            f"  Mark [{mid}]: CV shape={region.shape_type}, "
            f"bbox=(x={x}, y={y}, w={w}, h={h}), centroid=({cx},{cy})"
        )

    lines.append("")
    lines.append(_SNAPSHOT_RULES)
    lines.append(_SNAPSHOT_SCHEMA)
    return "\n".join(lines)


def _build_batch_contents(batch: list) -> list:
    """Build the API request contents for analysing N snapshots in one call.

    Interleaves per-snapshot image Parts with their CV mark context, prefixed
    by the shared schema and output format instructions.  The model is asked to
    return a single JSON OBJECT keyed by segment_id string — one request instead
    of one per snapshot, dramatically reducing the daily request count.
    """
    n = len(batch)
    seg_ids_str = ", ".join(str(kf.segment_id) for kf, _ in batch)

    contents: list = [
        # Schema + output format at the top so the model sees it before images
        f"TASK: Batch Whiteboard Snapshot Analysis ({n} snapshots in one API call)\n\n"
        f"Analyse each of the {n} whiteboard snapshots shown below.\n"
        f"Segment IDs to analyse: [{seg_ids_str}]\n\n"
        "OUTPUT FORMAT\n"
        "Return a single JSON OBJECT where each top-level key is the snapshot's\n"
        "segment_id as a STRING and each value is that snapshot's full analysis.\n"
        f"Your response MUST contain all {n} keys: [{seg_ids_str}]\n\n"
        "Example structure (IDs 3 and 7 shown):\n"
        '{\n'
        '  "3": { "mark_descriptions": {}, "connections": [], "board_state": "...", ... },\n'
        '  "7": { "mark_descriptions": {}, "connections": [], "board_state": "...", ... }\n'
        '}\n\n'
        f"PER-SNAPSHOT SCHEMA AND RULES:\n{_SNAPSHOT_RULES}\n\n{_SNAPSHOT_SCHEMA}\n",
    ]

    for kf, reg in batch:
        contents.append(
            f"\n--- SNAPSHOT {kf.segment_id} (timestamp: {kf.timestamp:.1f}s) ---"
        )
        contents.append(_image_part(Image.open(kf.image_path)))

        lines = [f"CV marks for snapshot {kf.segment_id}:"]
        for mark in sorted(kf.marks, key=lambda m: m["mark_id"]):
            mid = mark["mark_id"]
            region = reg.elements.get(mid)
            if region is None:
                continue
            x, y, w, h = region.bbox
            cx, cy = region.centroid
            lines.append(
                f"  Mark [{mid}]: CV shape={region.shape_type}, "
                f"bbox=(x={x}, y={y}, w={w}, h={h}), centroid=({cx},{cy})"
            )
        contents.append("\n".join(lines))

    contents.append(
        "\nReturn ONLY the JSON OBJECT with all snapshot analyses. "
        "No preamble, no markdown fences."
    )
    return contents


def _parse_snapshot(raw: dict) -> BoardSnapshot:
    """Parse VLM JSON response into a BoardSnapshot."""
    pmd_raw = raw.get("mark_descriptions", {})
    pmd: dict = {}
    for k, v in pmd_raw.items():
        try:
            pmd[int(k)] = v
        except (ValueError, TypeError):
            log.debug("_parse_snapshot: non-integer mark_descriptions key %r — skipped", k)
    return BoardSnapshot(
        segment_id=raw["_segment_id"],
        timestamp=raw["_timestamp"],
        mark_descriptions=pmd,
        connections=raw.get("connections", []),
        symbol_meanings=raw.get("symbol_meanings", []),
        groupings=raw.get("groupings", []),
        annotations=raw.get("annotations", []),
        cross_links=raw.get("cross_links", []),
        board_state=raw.get("board_state", ""),
        confidence=raw.get("confidence", "medium"),
        visibility_issues=raw.get("visibility_issues"),
    )


def analyse_snapshots(
    keyframes: list[KeyframeAnnotation],
    snapshot_registries: list[ElementRegistry],
    output_dir: Path,
) -> list[BoardSnapshot]:
    """Run batched VLM analysis on pen-lift keyframes for static IR.

    Snapshots are grouped into batches of config.SNAPSHOT_BATCH_SIZE (default 10)
    and sent in ONE API call per batch.  For 70 keyframes this is 7 calls instead
    of 70, keeping total usage well within the free-tier 20 req/day limit.

    The cache (snapshot_cache.json) is written after EACH batch.  If a run is
    interrupted mid-way (quota exhausted, crash, Ctrl-C), successfully analysed
    snapshots are preserved.  Segments with no data are left out of the cache so
    they will be retried on the next run — just re-run the pipeline; it picks up
    where it left off.

    Args:
        keyframes:           KeyframeAnnotation list from stage1 (one per pen-lift).
        snapshot_registries: Parallel list of ElementRegistry snapshots at each pen-lift.
        output_dir:          Directory for the snapshot cache.

    Returns:
        list[BoardSnapshot], one per keyframe, in the same order.
        Snapshots that could not be analysed are returned as empty BoardSnapshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "snapshot_cache.json"

    # Load partial cache keyed by segment_id
    cached_by_id: dict[int, dict] = {}
    if cache_path.exists():
        try:
            items = json.loads(cache_path.read_text())
            cached_by_id = {item["_segment_id"]: item for item in items}
            n_cached = sum(1 for kf in keyframes if kf.segment_id in cached_by_id)
            if n_cached == len(keyframes):
                log.info("Stage 2b: all %d snapshots loaded from cache", len(keyframes))
                return [_parse_snapshot(cached_by_id[kf.segment_id]) for kf in keyframes]
            log.info(
                "Stage 2b: partial cache — %d/%d snapshots already done, resuming",
                n_cached, len(keyframes),
            )
        except Exception as exc:
            log.warning("Stage 2b: cache unreadable (%s) — reprocessing all", exc)
            cached_by_id = {}

    # Only process segments not yet in the cache
    uncached = [
        (kf, reg)
        for kf, reg in zip(keyframes, snapshot_registries)
        if kf.segment_id not in cached_by_id
    ]

    if not uncached:
        # Shouldn't reach here after the check above, but be safe
        return [
            _parse_snapshot(cached_by_id.get(
                kf.segment_id, {"_segment_id": kf.segment_id, "_timestamp": kf.timestamp}
            ))
            for kf in keyframes
        ]

    batch_size = config.SNAPSHOT_BATCH_SIZE
    batches = [uncached[i: i + batch_size] for i in range(0, len(uncached), batch_size)]

    total_calls = 2 + len(batches)  # 2 for main video analysis + verification
    log.info(
        "Stage 2b: %d snapshots to analyse in %d batch(es) of up to %d "
        "(~%d total API calls this run)",
        len(uncached), len(batches), batch_size, total_calls,
    )

    client = _get_client()

    for batch_idx, batch in enumerate(batches):
        seg_ids = [kf.segment_id for kf, _ in batch]
        log.info("Stage 2b: batch %d/%d — segments %s", batch_idx + 1, len(batches), seg_ids)

        contents = _build_batch_contents(batch)

        try:
            text = _stream_generate(
                client, contents,
                f"snapshot batch {batch_idx + 1}/{len(batches)}",
            )
            batch_raw = _extract_json(text)

            # Normalise response: model may wrap in a "snapshots" key
            if isinstance(batch_raw, dict) and "snapshots" in batch_raw:
                inner = batch_raw["snapshots"]
                if isinstance(inner, dict):
                    batch_raw = inner

            # Normalise response: model may return an array instead of an object
            if isinstance(batch_raw, list):
                mapped: dict = {}
                for i, item in enumerate(batch_raw):
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("segment_id") or item.get("_segment_id")
                    if sid is None and i < len(batch):
                        sid = batch[i][0].segment_id
                    if sid is not None:
                        mapped[str(sid)] = item
                batch_raw = mapped

            # Extract per-snapshot results; only keep entries with real analysis data
            for kf, _ in batch:
                raw = batch_raw.get(str(kf.segment_id), {})
                if not isinstance(raw, dict):
                    raw = {}
                raw["_segment_id"] = kf.segment_id
                raw["_timestamp"] = kf.timestamp
                if raw.get("mark_descriptions") is not None or raw.get("board_state"):
                    cached_by_id[kf.segment_id] = raw
                else:
                    log.warning(
                        "Stage 2b: segment %d absent from batch response — will retry next run",
                        kf.segment_id,
                    )

        except Exception as exc:
            log.error(
                "Stage 2b: batch %d/%d failed (%s) — segments %s will be retried next run",
                batch_idx + 1, len(batches), exc, seg_ids,
            )

        # Persist progress after every batch (crash / quota safety)
        all_cached = [
            cached_by_id[kf.segment_id]
            for kf in keyframes
            if kf.segment_id in cached_by_id
        ]
        cache_path.write_text(json.dumps(all_cached, indent=2))
        log.info(
            "Stage 2b: %d/%d snapshots saved after batch %d/%d",
            len(all_cached), len(keyframes), batch_idx + 1, len(batches),
        )

    # Return all keyframes; segments without analysis become empty BoardSnapshots
    return [
        _parse_snapshot(cached_by_id.get(
            kf.segment_id, {"_segment_id": kf.segment_id, "_timestamp": kf.timestamp}
        ))
        for kf in keyframes
    ]
