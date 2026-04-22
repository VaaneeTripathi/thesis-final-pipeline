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
import json
import logging
import shutil
import tempfile
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
import numpy as np

from pipeline.models import (
    BoardSnapshot, DetectedRegion, ElementRegistry,
    KeyframeAnnotation, TemporalSegment, VLMOperation,
)
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
        if snap is None:
            log.warning("S emission skipped (%s): empty registry — no marks to serialize", label)
            return
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


# Cache files are preserved across runs — delete manually to force a fresh run.
# stage1_cache.json covers the CV stage; vlm_cache/snapshot_cache cover VLM calls.
_CACHE_FILES = {"vlm_cache.json", "snapshot_cache.json", "stage1_cache.json"}

_STAGE1_CACHE = "stage1_cache.json"


def _clean_output_dir(output_dir: Path) -> None:
    """Wipe output_dir and recreate it, preserving cache files and keyframes/.

    The keyframes/ directory is moved to a temp location and restored after
    the wipe — this avoids reading all PNGs into memory while still keeping
    them alongside their stage1_cache.json metadata.
    """
    saved: dict[str, str] = {}
    tmp_keyframes: Path | None = None

    if output_dir.exists():
        for fname in _CACHE_FILES:
            p = output_dir / fname
            if p.exists():
                saved[fname] = p.read_text(encoding="utf-8")

        # Preserve keyframes/ only when the stage 1 cache is also present
        kf_dir = output_dir / "keyframes"
        if kf_dir.exists() and (output_dir / _STAGE1_CACHE).exists():
            tmp_parent = Path(tempfile.mkdtemp())
            tmp_keyframes = tmp_parent / "keyframes"
            shutil.move(str(kf_dir), str(tmp_keyframes))

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, content in saved.items():
        (output_dir / fname).write_text(content, encoding="utf-8")

    if tmp_keyframes and tmp_keyframes.exists():
        shutil.move(str(tmp_keyframes), str(output_dir / "keyframes"))
        shutil.rmtree(str(tmp_keyframes.parent))

    preserved = sorted(saved) + (["keyframes/"] if tmp_keyframes else [])
    if preserved:
        log.info("Output directory cleaned (preserved: %s)", ", ".join(preserved))
    else:
        log.info("Output directory cleaned")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _save_stage1_cache(
    registry: ElementRegistry,
    segments: list,
    registry_snapshots: list,
    keyframe_annotations: list,
    output_dir: Path,
) -> None:
    """Persist stage 1 outputs to stage1_cache.json."""

    def _reg_to_dict(reg: ElementRegistry) -> dict:
        return {
            "next_id": reg.next_id,
            "elements": {
                str(k): {
                    "mark_id": v.mark_id,
                    "bbox": list(v.bbox),
                    "shape_type": v.shape_type,
                    "centroid": list(v.centroid),
                    "first_seen": v.first_seen,
                }
                for k, v in reg.elements.items()
            },
        }

    data = {
        "registry": _reg_to_dict(registry),
        "segments": [
            {
                "segment_id": s.segment_id,
                "timestamp_start": s.timestamp_start,
                "timestamp_end": s.timestamp_end,
                "segment_type": s.segment_type,
                "delta_magnitude": s.delta_magnitude,
            }
            for s in segments
        ],
        "registry_snapshots": [_reg_to_dict(r) for r in registry_snapshots],
        "keyframe_annotations": [
            {
                "segment_id": k.segment_id,
                "timestamp": k.timestamp,
                "image_path": str(k.image_path),
                "marks": k.marks,
            }
            for k in keyframe_annotations
        ],
    }
    (output_dir / _STAGE1_CACHE).write_text(
        json.dumps(data, indent=2, cls=_NumpyEncoder), encoding="utf-8"
    )
    log.info("Stage 1: cache written → %s", _STAGE1_CACHE)


def _load_stage1_cache(
    output_dir: Path,
) -> tuple | None:
    """Load stage 1 cache if present and all keyframe PNGs still exist.

    Returns (registry, segments, registry_snapshots, keyframe_annotations)
    or None if the cache is absent or stale.
    """
    cache_path = output_dir / _STAGE1_CACHE
    if not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text(encoding="utf-8"))

    def _dict_to_reg(d: dict) -> ElementRegistry:
        reg = ElementRegistry()
        reg.next_id = d["next_id"]
        for k, v in d["elements"].items():
            reg.elements[int(k)] = DetectedRegion(
                mark_id=v["mark_id"],
                bbox=tuple(v["bbox"]),
                shape_type=v["shape_type"],
                centroid=tuple(v["centroid"]),
                contour=np.array([]),   # not used downstream
                first_seen=v["first_seen"],
            )
        return reg

    registry = _dict_to_reg(data["registry"])

    segments = [
        TemporalSegment(
            segment_id=s["segment_id"],
            timestamp_start=s["timestamp_start"],
            timestamp_end=s["timestamp_end"],
            segment_type=s["segment_type"],
            delta_magnitude=s["delta_magnitude"],
            keyframe_before=np.array([]),  # not used downstream
            keyframe_after=np.array([]),   # not used downstream
        )
        for s in data["segments"]
    ]

    registry_snapshots = [_dict_to_reg(r) for r in data["registry_snapshots"]]

    keyframe_annotations = []
    for k in data["keyframe_annotations"]:
        img_path = Path(k["image_path"])
        if not img_path.exists():
            log.warning(
                "Stage 1 cache stale: keyframe PNG missing (%s) — re-running stage 1",
                img_path,
            )
            return None
        keyframe_annotations.append(
            KeyframeAnnotation(
                segment_id=k["segment_id"],
                timestamp=k["timestamp"],
                image_path=img_path,
                marks=k["marks"],
            )
        )

    log.info(
        "Stage 1: loaded from cache (%d segments, %d keyframes)",
        len(segments), len(keyframe_annotations),
    )
    return registry, segments, registry_snapshots, keyframe_annotations


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

    # Stage 0: ingest
    cap, ingest_data = stage0_ingest.ingest(video_path)

    # Stage 1: CV temporal segmentation + keyframe annotation
    stage1_result = _load_stage1_cache(output_dir)
    if stage1_result is not None:
        cap.release()
        registry, segments, registry_snapshots, keyframe_annotations = stage1_result
    else:
        try:
            registry, segments, registry_snapshots, keyframe_annotations = stage1_cv.run(
                cap, ingest_data, output_dir,
            )
        finally:
            cap.release()
        _save_stage1_cache(registry, segments, registry_snapshots, keyframe_annotations, output_dir)
        log.info(
            "Stage 1: %d segments, %d final marks, %d keyframes",
            len(segments), len(registry.elements), len(keyframe_annotations),
        )

    # Stage 2a: VLM operation analysis (whole video) 
    operations = stage2_vlm.run(
        video_path=video_path,
        registry=registry,
        output_dir=output_dir,
        keyframes=keyframe_annotations,
    )
    log.info("Stage 2a: %d VLM operations", len(operations))

    if not operations:
        log.warning("No VLM operations — pipeline output will be empty.")

    #  Stage 2b: VLM snapshot analysis (per-keyframe, for static IR) 
    board_snapshots = stage2_vlm.analyse_snapshots(
        keyframes=keyframe_annotations,
        snapshot_registries=registry_snapshots,
        output_dir=output_dir,
    )
    log.info("Stage 2b: %d board snapshots analysed", len(board_snapshots))

    # Stages 3–5: Mealy machine + IR assembly 
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

    # Stage 6: validate + semantic deltas
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

    # Stage 7: tactile rendering (graceful skip if not installed) 
    tactile_dirs = stage7_tactile.run(output_dir)
    if tactile_dirs:
        log.info("Stage 7: %d snapshot(s) rendered tactilely", len(tactile_dirs))
    else:
        log.info("Stage 7: tactile rendering skipped or no snapshots to render")

    # Stage 8: transcript 
    txt_path, json_path = stage8_transcript.generate(operation_doc, output_dir)
    log.info("Stage 8: transcript written → %s", txt_path.name)

   
    wall_elapsed = time.time() - wall_start
    log.info(
        "Pipeline complete in %.1fs — valid=%s, errors=%d",
        wall_elapsed, report.is_valid, len(report.all_errors),
    )

    if not report.is_valid:
        for err in report.all_errors:
            log.error("  %s", err)
