"""End-to-end pipeline orchestrator.

Wires all stages together and drives the Mealy machine to determine
when to emit static IR snapshots (S) and operation entries (P).

Data flow:
  Stage 0 → ingest video metadata + ROI
  Stage 1 → CV temporal segmentation + annotated keyframe PNGs
  Stage 2a → VLM operation analysis (whole video + keyframes)
  Stage 2b → VLM snapshot analysis (per-keyframe, for static IR)
  Stage 3  → Mealy machine drives S/P emissions
  Stage 4  → Static IR per S emission (CV geometry + VLM snapshot semantics)
  Stage 5  → Operation entries per P emission → assembled into one document
  Stage 6  → Schema validation + semantic deltas
  Stage 7  → Tactile rendering (graceful skip if unavailable)
  Stage 8  → Transcript generation

Usage:
    from pipeline.pipeline import run
    run(Path("lecture.mp4"), Path("outputs/"))
"""
from __future__ import annotations
import datetime
import logging
import shutil
import time
from pathlib import Path

from pipeline import config
from pipeline import (
    stage0_ingest,
    stage1_cv,
    stage2_vlm,
    stage4_static_ir,
    stage5_operations,
    stage6_validate,
    stage7_tactile,
    stage8_transcript,
)
from pipeline.models import BoardSnapshot, ElementRegistry, VLMOperation
from pipeline.stage3_mealy import (
    MealyMachine,
    OMEGA, TAU,
    OUT_S, OUT_P, OUT_SP,
)

log = logging.getLogger(__name__)


def _parse_mmss(ts: str) -> float:
    """MM:SS or MM:SS.mmm → float seconds."""
    try:
        parts = ts.split(":")
        m = int(parts[0])
        s_parts = parts[1].split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return m * 60 + s + ms / 1000
    except Exception:
        return 0.0


def _seconds_to_mmss(seconds: float) -> str:
    total_s = int(seconds)
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


def _find_matching_segment(vlm_op: VLMOperation, segments):
    """Find the TemporalSegment whose midpoint is closest to the VLM operation midpoint."""
    op_start = _parse_mmss(vlm_op.timestamp_start)
    op_end = _parse_mmss(vlm_op.timestamp_end)
    op_mid = (op_start + op_end) / 2

    best = None
    best_dist = float("inf")
    for seg in segments:
        seg_mid = (seg.timestamp_start + seg.timestamp_end) / 2
        dist = abs(op_mid - seg_mid)
        if dist < best_dist:
            best_dist = dist
            best = seg

    return best if best_dist <= 30.0 else None


def _get_registry(seg, segments, registry_snapshots: list[ElementRegistry],
                  final_registry: ElementRegistry) -> ElementRegistry:
    if seg is None:
        return final_registry
    try:
        idx = list(segments).index(seg)
        return registry_snapshots[idx] if idx < len(registry_snapshots) else final_registry
    except ValueError:
        return final_registry


def _get_board_snapshot(
    seg,
    snapshot_by_seg_id: dict[int, BoardSnapshot],
    fallback: BoardSnapshot | None,
) -> BoardSnapshot | None:
    if seg is None:
        return fallback
    return snapshot_by_seg_id.get(seg.segment_id, fallback)


def _process_operations(
    operations: list[VLMOperation],
    segments,
    registry_snapshots: list[ElementRegistry],
    board_snapshots: list[BoardSnapshot],
    final_registry: ElementRegistry,
    video_name: str,
    video_duration: float,
    fps: float,
    analysis_timestamp: str,
    time_taken_str: str,
) -> tuple[list[dict], dict, MealyMachine]:
    """Drive the Mealy machine over all VLM operations.

    Returns:
        snapshots      — list of static IR dicts (one per Mealy S emission)
        operation_doc  — single aggregated operation IR document
        mealy          — MealyMachine instance (history attached)
    """
    mealy = MealyMachine()
    snapshots: list[dict] = []
    op_entries: list[dict] = []

    # Index board snapshots by segment_id for O(1) lookup
    snapshot_by_seg_id: dict[int, BoardSnapshot] = {
        bs.segment_id: bs for bs in board_snapshots
    }
    last_board_snapshot = board_snapshots[-1] if board_snapshots else None

    def _emit_s(registry: ElementRegistry, board_snap: BoardSnapshot | None, label: str) -> None:
        if board_snap is None:
            log.warning("S emission skipped (%s): no board snapshot available", label)
            return
        snap = stage4_static_ir.build(registry, board_snap, time_taken=time_taken_str)
        snapshots.append(snap)
        log.debug("Emitted S (%s)", label)

    prev_op_end_s = 0.0

    for i, vlm_op in enumerate(operations):
        matching_seg = _find_matching_segment(vlm_op, segments)
        cur_registry = _get_registry(matching_seg, segments, registry_snapshots, final_registry)
        cur_snapshot = _get_board_snapshot(matching_seg, snapshot_by_seg_id, last_board_snapshot)

        # Idle timeout gap before this operation 
        op_start_s = _parse_mmss(vlm_op.timestamp_start)
        if op_start_s - prev_op_end_s > config.IDLE_TIMEOUT:
            tau_out = mealy.step(TAU, context={"gap_seconds": op_start_s - prev_op_end_s})
            if OUT_S in tau_out:
                _emit_s(cur_registry, cur_snapshot, f"TAU gap before op {vlm_op.operation_id}")

        #  Feed sigma (the operation type) 
        sigma_out = mealy.step(
            vlm_op.operation_type,
            context={"op_id": vlm_op.operation_id, "type": vlm_op.operation_type},
        )

        if OUT_P in sigma_out:
            entry = stage5_operations.build_entry(vlm_op=vlm_op, segment=matching_seg)
            op_entries.append(entry)
            log.debug("Emitted P for op %d (%s)", vlm_op.operation_id, vlm_op.operation_type)

        if OUT_S in sigma_out:
            # Pre-operation snapshot (board state just before this op)
            _emit_s(cur_registry, cur_snapshot, f"pre-op {vlm_op.operation_id}")

        #  Feed OMEGA (pen-lift completing the operation) 
        omega_out = mealy.step(
            OMEGA,
            context={
                "op_id": vlm_op.operation_id,
                "seg_id": matching_seg.segment_id if matching_seg else None,
            },
        )

        if OUT_S in omega_out:
            # Post-operation snapshot (board state after this op completes)
            _emit_s(cur_registry, cur_snapshot, f"post-op {vlm_op.operation_id}")

        prev_op_end_s = _parse_mmss(vlm_op.timestamp_end)

    #  Trailing TAU after the last operation 
    if operations:
        tail_gap = video_duration - prev_op_end_s
        if tail_gap > config.IDLE_TIMEOUT:
            tau_out = mealy.step(TAU, context={"tail_gap_seconds": tail_gap})
            if OUT_S in tau_out:
                _emit_s(final_registry, last_board_snapshot, "trailing TAU")

    #  Assemble single operation document from collected entries 
    operation_doc = stage5_operations.assemble_document(
        entries=op_entries,
        all_operations=operations,
        video_name=video_name,
        video_duration=video_duration,
        fps=fps,
        time_taken=time_taken_str,
        analysis_timestamp=analysis_timestamp,
    )

    return snapshots, operation_doc, mealy


# Cache files are preserved across runs — they represent expensive VLM API
# calls. Delete them manually to force a fresh analysis.
_CACHE_FILES = {"vlm_cache.json", "snapshot_cache.json"}


def _clean_output_dir(output_dir: Path) -> None:
    """Wipe output_dir and recreate it, preserving any VLM cache files.

    Called at the start of every run so stale outputs from a previous run
    never contaminate the current one.
    """
    saved: dict[str, str] = {}
    if output_dir.exists():
        for fname in _CACHE_FILES:
            p = output_dir / fname
            if p.exists():
                saved[fname] = p.read_text(encoding="utf-8")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, content in saved.items():
        (output_dir / fname).write_text(content, encoding="utf-8")

    if saved:
        log.info("Output directory cleaned (caches preserved: %s)", ", ".join(sorted(saved)))
    else:
        log.info("Output directory cleaned")


def run(video_path: Path, output_dir: Path) -> None:
    """Run the full pipeline: video → accessible IR + tactile + transcript.

    Outputs written to output_dir/:
      snapshots/snapshot_NNNN.json    — static IR per Mealy S emission
      operations/operations.json      — aggregated operation IR (one per video)
      deltas/delta_NNNN.json          — semantic graph deltas between snapshots
      keyframes/keyframe_NNNN.png     — annotated keyframe PNGs (from Stage 1)
      tactile/snapshot_NNNN/          — tactile rendering outputs (if available)
      transcript.txt / transcript.json
      validation_report.json
    """
    wall_start = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("=== Pipeline start: %s ===", video_path.name)
    _clean_output_dir(output_dir)

    # --- Stage 0: ingest ---
    cap, ingest_data = stage0_ingest.ingest(video_path)

    try:
        # --- Stage 1: CV temporal segmentation + keyframe annotation ---
        registry, segments, registry_snapshots, keyframe_annotations = stage1_cv.run(
            cap, ingest_data, output_dir,
        )
    finally:
        cap.release()

    log.info(
        "Stage 1: %d segments, %d final marks, %d keyframes",
        len(segments), len(registry.elements), len(keyframe_annotations),
    )

    # --- Stage 2a: VLM operation analysis (whole video) ---
    operations = stage2_vlm.run(
        video_path=video_path,
        registry=registry,
        output_dir=output_dir,
        keyframes=keyframe_annotations,
    )
    log.info("Stage 2a: %d VLM operations", len(operations))

    if not operations:
        log.warning("No VLM operations — pipeline output will be empty.")

    # --- Stage 2b: VLM snapshot analysis (per-keyframe, for static IR) ---
    board_snapshots = stage2_vlm.analyse_snapshots(
        keyframes=keyframe_annotations,
        snapshot_registries=registry_snapshots,
        output_dir=output_dir,
    )
    log.info("Stage 2b: %d board snapshots analysed", len(board_snapshots))

    # --- Stages 3–5: Mealy machine + IR assembly ---
    analysis_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    time_taken_str = _seconds_to_mmss(time.time() - wall_start)

    snapshots, operation_doc, mealy = _process_operations(
        operations=operations,
        segments=segments,
        registry_snapshots=registry_snapshots,
        board_snapshots=board_snapshots,
        final_registry=registry,
        video_name=video_path.name,
        video_duration=ingest_data.duration,
        fps=ingest_data.fps,
        analysis_timestamp=analysis_timestamp,
        time_taken_str=time_taken_str,
    )

    n_ops_out = len(operation_doc.get("analysis", {}).get("operations", []))
    log.info("Stages 3–5: %d snapshots, %d operation entries", len(snapshots), n_ops_out)

    # --- Stage 6: validate + semantic deltas ---
    report, deltas = stage6_validate.run(
        snapshots=snapshots,
        operation_doc=operation_doc,
        mealy_history=mealy.history,
        output_dir=output_dir,
    )
    log.info(
        "Stage 6: %d deltas, valid=%s, errors=%d",
        len(deltas), report.is_valid, len(report.all_errors),
    )

    # --- Stage 7: tactile rendering (graceful skip if not installed) ---
    tactile_dirs = stage7_tactile.run(output_dir)
    if tactile_dirs:
        log.info("Stage 7: %d snapshot(s) rendered tactilely", len(tactile_dirs))
    else:
        log.info("Stage 7: tactile rendering skipped or no snapshots to render")

    # --- Stage 8: transcript ---
    txt_path, json_path = stage8_transcript.generate(operation_doc, output_dir)
    log.info("Stage 8: transcript written → %s", txt_path.name)

    # --- Final summary ---
    wall_elapsed = time.time() - wall_start
    log.info(
        "=== Pipeline complete in %.1fs — valid=%s, errors=%d ===",
        wall_elapsed, report.is_valid, len(report.all_errors),
    )

    if not report.is_valid:
        for err in report.all_errors:
            log.error("  %s", err)
