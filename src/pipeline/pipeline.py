"""End-to-end pipeline orchestrator.

Wires all stages together and drives the Mealy machine to determine
when to emit static IR snapshots (S) and operation IRs (P).

Usage:
    from pipeline.pipeline import run
    run(Path("lecture.mp4"), Path("outputs/"))
"""
from __future__ import annotations
import datetime
import json
import logging
import time
from pathlib import Path

from pipeline import config
from pipeline import stage0_ingest, stage1_cv, stage2_vlm, stage6_validate
from pipeline import stage4_static_ir, stage5_operations
from pipeline.models import ElementRegistry, VLMOperation
from pipeline.stage3_mealy import (
    MealyMachine,
    OMEGA, TAU,
    OUT_S, OUT_P, OUT_SP,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_mmss(ts: str) -> float:
    """MM:SS or MM:SS.mmm -> float seconds."""
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


# ---------------------------------------------------------------------------
# Segment matching
# ---------------------------------------------------------------------------

def _find_matching_segment(vlm_op: VLMOperation, segments):
    """Find the TemporalSegment whose time window best overlaps the VLM operation."""
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

    # Only match if within 30s — otherwise return None
    if best_dist > 30.0:
        return None
    return best


# ---------------------------------------------------------------------------
# Mealy event processing
# ---------------------------------------------------------------------------

def _process_operations(
    operations: list[VLMOperation],
    segments,
    registry_snapshots: list[ElementRegistry],
    final_registry: ElementRegistry,
    video_name: str,
    video_duration: float,
    analysis_timestamp: str,
    time_taken_str: str,
) -> tuple[list[dict], list[dict], list[list[dict]], MealyMachine]:
    """Drive the Mealy machine over all VLM operations.

    Returns (snapshots, operation_irs, patches, mealy_machine).
    """
    mealy = MealyMachine()
    snapshots: list[dict] = []
    operation_irs: list[dict] = []

    def _get_registry(seg) -> ElementRegistry:
        if seg is None:
            return final_registry
        try:
            idx = list(segments).index(seg)
            return registry_snapshots[idx] if idx < len(registry_snapshots) else final_registry
        except ValueError:
            return final_registry

    prev_op_end_s = 0.0

    for i, vlm_op in enumerate(operations):
        matching_seg = _find_matching_segment(vlm_op, segments)
        current_registry = _get_registry(matching_seg)
        ops_so_far = operations[: i + 1]

        # --- Idle timeout between operations? ---
        op_start_s = _parse_mmss(vlm_op.timestamp_start)
        if op_start_s - prev_op_end_s > config.IDLE_TIMEOUT:
            tau_out = mealy.step(TAU, context={"gap_seconds": op_start_s - prev_op_end_s})
            if OUT_S in tau_out:
                snap = stage4_static_ir.build(
                    registry=current_registry,
                    operations=operations[:i],
                    snapshot_timestamp=prev_op_end_s + config.IDLE_TIMEOUT,
                    video_name=video_name,
                    time_taken=time_taken_str,
                )
                snapshots.append(snap)
                log.debug("TAU snapshot at %.1fs", prev_op_end_s + config.IDLE_TIMEOUT)

        # --- Feed sigma (the VLM operation type) ---
        sigma_out = mealy.step(
            vlm_op.operation_type,
            context={"op_id": vlm_op.operation_id, "type": vlm_op.operation_type},
        )

        if OUT_P in sigma_out:
            op_ir = stage5_operations.build(
                vlm_op=vlm_op,
                segment=matching_seg,
                video_name=video_name,
                video_duration=video_duration,
                total_operations=len(operations),
                all_operations=operations,
                time_taken=time_taken_str,
                analysis_timestamp=analysis_timestamp,
            )
            operation_irs.append(op_ir)
            log.debug("Emitted P for op %d (%s)", vlm_op.operation_id, vlm_op.operation_type)

        if OUT_S in sigma_out:
            # S in sigma output = snapshot of state BEFORE this operation
            snap = stage4_static_ir.build(
                registry=current_registry,
                operations=operations[:i],
                snapshot_timestamp=_parse_mmss(vlm_op.timestamp_start),
                video_name=video_name,
                time_taken=time_taken_str,
            )
            snapshots.append(snap)
            log.debug("Emitted S (pre-op) for op %d", vlm_op.operation_id)

        # --- Feed OMEGA (pen-lift after the operation) ---
        omega_out = mealy.step(
            OMEGA,
            context={"op_id": vlm_op.operation_id, "seg_id": matching_seg.segment_id if matching_seg else None},
        )

        if OUT_S in omega_out:
            snap = stage4_static_ir.build(
                registry=current_registry,
                operations=ops_so_far,
                snapshot_timestamp=_parse_mmss(vlm_op.timestamp_end),
                video_name=video_name,
                time_taken=time_taken_str,
            )
            snapshots.append(snap)
            log.debug("Emitted S (post-op) for op %d", vlm_op.operation_id)

        prev_op_end_s = _parse_mmss(vlm_op.timestamp_end)

    # --- Trailing TAU after the last operation ---
    if operations:
        tail_gap = video_duration - prev_op_end_s
        if tail_gap > config.IDLE_TIMEOUT:
            tau_out = mealy.step(TAU, context={"tail_gap_seconds": tail_gap})
            if OUT_S in tau_out:
                snap = stage4_static_ir.build(
                    registry=final_registry,
                    operations=operations,
                    snapshot_timestamp=video_duration,
                    video_name=video_name,
                    time_taken=time_taken_str,
                )
                snapshots.append(snap)

    # --- Compute patches between consecutive snapshots ---
    patches = [
        stage6_validate.compute_patch(snapshots[j], snapshots[j + 1])
        for j in range(len(snapshots) - 1)
    ]

    return snapshots, operation_irs, patches, mealy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(video_path: Path, output_dir: Path) -> None:
    """Run the full pipeline: video -> IR schemas.

    Outputs written to output_dir/:
      snapshots/snapshot_NNNN.json   — static IR per Mealy S emission
      operations/operation_NNNN.json — operation IR per Mealy P emission
      patches/patch_NNNN.json        — RFC 6902 patches between snapshots
      marked_video.mp4               — SoM-marked video (from Stage 1)
      vlm_cache.json                 — cached VLM response (from Stage 2)
      validation_report.json         — Stage 6 validation results
    """
    wall_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("=== Pipeline start: %s ===", video_path.name)

    # --- Stage 0: ingest ---
    cap, ingest_data = stage0_ingest.ingest(video_path)

    try:
        # --- Stage 1: CV + SoM ---
        registry, segments, registry_snapshots, marked_video_path = stage1_cv.run(
            cap, ingest_data, output_dir
        )
    finally:
        cap.release()

    log.info(
        "Stage 1: %d segments, %d final marks, marked video -> %s",
        len(segments), len(registry.elements), marked_video_path.name,
    )

    # --- Stage 2: VLM ---
    operations = stage2_vlm.run(marked_video_path, registry, output_dir)
    log.info("Stage 2: %d operations from VLM", len(operations))

    if not operations:
        log.warning("No VLM operations returned — pipeline output will be empty.")

    # --- Stages 3–5: Mealy + IR generation ---
    analysis_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    elapsed_stage2 = time.time() - wall_start
    time_taken_str = _seconds_to_mmss(elapsed_stage2)

    snapshots, operation_irs, patches, mealy = _process_operations(
        operations=operations,
        segments=segments,
        registry_snapshots=registry_snapshots,
        final_registry=registry,
        video_name=video_path.name,
        video_duration=ingest_data.duration,
        analysis_timestamp=analysis_timestamp,
        time_taken_str=time_taken_str,
    )

    log.info(
        "Stages 3–5: %d snapshots, %d operation IRs, %d patches",
        len(snapshots), len(operation_irs), len(patches),
    )

    # --- Stage 6: validate + write ---
    report = stage6_validate.run(
        snapshots=snapshots,
        operation_irs=operation_irs,
        patches=patches,
        mealy_history=mealy.history,
        output_dir=output_dir,
    )

    wall_elapsed = time.time() - wall_start
    log.info(
        "=== Pipeline complete in %.1fs — valid=%s, errors=%d ===",
        wall_elapsed, report.is_valid, len(report.all_errors),
    )

    if not report.is_valid:
        for err in report.all_errors:
            log.error("  %s", err)
