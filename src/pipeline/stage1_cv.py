"""Stage 1: CV processing, temporal segmentation, and keyframe annotation.

Single pass through the ROI-cropped video:
  1. GMM background subtraction → foreground ratio per frame
  2. SSIM between sampled frames → perceptual change magnitude
  3. Hysteresis thresholding → activity / idle temporal segmentation
  4. On pen-lift: contour detection + noise filtering + shape classification
     → ElementRegistry update
  5. On pen-lift: draw SoM marks on the stable keyframe_after, write PNG

No marked video is produced. SoM marks are applied only to the still
keyframe at each pen-lift, keeping the original video clean for the VLM.

Outputs:
  - ElementRegistry         — final state after all pen-lifts
  - list[TemporalSegment]   — one per activity window
  - list[ElementRegistry]   — registry snapshot at each pen-lift (parallel to segments)
  - list[KeyframeAnnotation] — one annotated PNG per pen-lift
"""
from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pipeline import config
from pipeline.models import (
    DetectedRegion,
    ElementRegistry,
    KeyframeAnnotation,
    TemporalSegment,
)
from pipeline.stage0_ingest import VideoIngest

log = logging.getLogger(__name__)

_CONNECTION_SHAPES = {"arrow", "line", "connection"}


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a BGR ROI frame to gray, binary, and edge images."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.medianBlur(gray, 5))

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12,
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    return gray, binary, edges


def _classify_shape(contour: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Classify a closed contour as rectangle, diamond, oval, circle, triangle, or other."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "other"

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    n = len(approx)

    area = cv2.contourArea(contour)
    bbox_area = w * h if w > 0 and h > 0 else 1
    extent = area / bbox_area
    aspect = float(w) / h if h > 0 else 1.0
    circularity = 4.0 * np.pi * area / (peri * peri)

    if n == 3:
        return "triangle"
    if n == 4:
        return "rectangle" if extent >= 0.65 else "diamond"
    if circularity >= 0.6 or n >= 7:
        if extent > 0.5:
            return "circle" if 0.85 <= aspect <= 1.15 else "oval"
    return "other"


def _passes_noise_filters(
    contour: np.ndarray,
    x: int, y: int, w: int, h: int,
    frame_shape: tuple[int, int],
) -> bool:
    """Return True if the detection passes solidity and edge-margin filters.

    Edge margin: skip contours whose bounding box touches within EDGE_MARGIN
    pixels of the frame boundary — these are typically partial shapes at the
    ROI edge, not real whiteboard elements.

    Solidity: contour_area / convex_hull_area — blobs far below 1.0 are noisy
    irregular fragments, not clean shapes.
    """
    frame_h, frame_w = frame_shape
    margin = config.EDGE_MARGIN

    if (x < margin or y < margin or
            x + w > frame_w - margin or y + h > frame_h - margin):
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = cv2.contourArea(contour) / hull_area
        if solidity < config.MIN_SOLIDITY:
            return False

    return True


def _apply_marks_cap(detections: list[dict]) -> list[dict]:
    """Keep only the top MAX_MARKS detections ranked by contour area."""
    if len(detections) <= config.MAX_MARKS:
        return detections
    ranked = sorted(detections, key=lambda d: cv2.contourArea(d["contour"]), reverse=True)
    log.debug("Marks cap applied: %d → %d detections", len(detections), config.MAX_MARKS)
    return ranked[: config.MAX_MARKS]


def _detect_nodes(binary: np.ndarray, gray: np.ndarray) -> list[dict]:
    """Detect closed-shape nodes via contours, with noise filters applied."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_shape = binary.shape[:2]
    detections: list[dict] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h

        if not (config.MIN_CONTOUR_AREA <= bbox_area <= config.MAX_CONTOUR_AREA):
            continue
        if w < config.MIN_NODE_WIDTH or h < config.MIN_NODE_HEIGHT:
            continue

        extent = area / bbox_area if bbox_area > 0 else 0
        if extent < 0.08:
            continue

        if not _passes_noise_filters(contour, x, y, w, h, frame_shape):
            continue

        shape_type = _classify_shape(contour, x, y, w, h)

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        detections.append({
            "bbox": (x, y, w, h),
            "shape_type": shape_type,
            "contour": contour,
            "centroid": (cx, cy),
        })

    return detections


def _detect_connections(
    edges: np.ndarray,
    node_bboxes: list[tuple[int, int, int, int]],
) -> list[dict]:
    """Detect line/arrow connections via HoughLinesP.

    Lines whose both endpoints fall within the same node bbox are discarded
    (internal marks). All others connecting distinct regions are kept.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return []

    frame_shape = edges.shape[:2]
    detections: list[dict] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if _point_in_any_bbox(x1, y1, node_bboxes) and _point_in_any_bbox(x2, y2, node_bboxes):
            b1 = _bbox_containing(x1, y1, node_bboxes)
            b2 = _bbox_containing(x2, y2, node_bboxes)
            if b1 == b2:
                continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bx = min(x1, x2)
        by = min(y1, y2)
        bw = abs(x2 - x1) + 1
        bh = abs(y2 - y1) + 1
        contour = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32)

        # Apply edge margin to connection midpoint
        if not _passes_noise_filters(contour, bx, by, bw, bh, frame_shape):
            continue

        detections.append({
            "bbox": (bx, by, bw, bh),
            "shape_type": "connection",
            "contour": contour,
            "centroid": (cx, cy),
        })

    return detections


def _point_in_any_bbox(
    px: int, py: int, bboxes: list[tuple[int, int, int, int]]
) -> bool:
    for x, y, w, h in bboxes:
        if x <= px <= x + w and y <= py <= y + h:
            return True
    return False


def _bbox_containing(
    px: int, py: int, bboxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    for bbox in bboxes:
        x, y, w, h = bbox
        if x <= px <= x + w and y <= py <= y + h:
            return bbox
    return None


def _draw_som_marks(frame: np.ndarray, registry: ElementRegistry) -> np.ndarray:
    """Overlay numbered SoM labels and bounding boxes on a still frame.

    Nodes get green boxes. Connections get red boxes. Used only for
    keyframe PNGs — not applied to video frames.
    """
    out = frame.copy()
    for mark_id, region in registry.elements.items():
        x, y, w, h = region.bbox
        is_conn = region.shape_type in _CONNECTION_SHAPES
        color = config.MARK_CONNECTION_COLOR if is_conn else config.MARK_NODE_COLOR

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label = str(mark_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = config.MARK_FONT_SCALE
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        lx = max(x, 0)
        ly = max(y - 5, th + baseline)
        cv2.rectangle(out, (lx, ly - th - baseline), (lx + tw + 2, ly + baseline), (0, 0, 0), -1)
        cv2.putText(out, label, (lx + 1, ly), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def _write_keyframe(
    keyframe_after: np.ndarray,
    registry: ElementRegistry,
    segment_id: int,
    timestamp: float,
    output_dir: Path,
) -> KeyframeAnnotation:
    """Draw SoM marks on keyframe_after and write to output_dir/keyframes/.

    Returns a KeyframeAnnotation with the PNG path and mark metadata.
    The annotated image is sent to the VLM alongside the clean original video.
    """
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    annotated = _draw_som_marks(keyframe_after, registry)
    image_path = keyframes_dir / f"keyframe_{segment_id:04d}.png"
    cv2.imwrite(str(image_path), annotated)

    marks = [
        {
            "mark_id": region.mark_id,
            "shape_type": region.shape_type,
            "centroid": region.centroid,
            "bbox": region.bbox,
        }
        for region in registry.elements.values()
    ]

    return KeyframeAnnotation(
        segment_id=segment_id,
        timestamp=timestamp,
        image_path=image_path,
        marks=marks,
    )


def _classify_segment_type(before: np.ndarray, after: np.ndarray) -> str:
    """Classify an activity segment as 'activity' or 'erasure'.

    Erasure = ink density dropped significantly from before to after.
    """
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    density_before = np.mean(gray_before < 128)
    density_after = np.mean(gray_after < 128)

    if density_before > 0.01 and density_after < density_before * 0.70:
        return "erasure"
    return "activity"


def _snapshot_registry(registry: ElementRegistry) -> ElementRegistry:
    """Frozen copy of the registry at a pen-lift moment.

    Creates new DetectedRegion objects so centroid/bbox/shape are immutable.
    Numpy contour arrays are shared (read-only after creation).
    """
    snap = ElementRegistry(next_id=registry.next_id)
    for mark_id, region in registry.elements.items():
        snap.elements[mark_id] = DetectedRegion(
            mark_id=region.mark_id,
            bbox=region.bbox,
            shape_type=region.shape_type,
            centroid=region.centroid,
            contour=region.contour,
            first_seen=region.first_seen,
        )
    return snap


def run(
    cap: cv2.VideoCapture,
    ingest_data: VideoIngest,
    output_dir: Path,
) -> tuple[ElementRegistry, list[TemporalSegment], list[ElementRegistry], list[KeyframeAnnotation]]:
    """Process the full video: temporal segmentation + pen-lift keyframe annotation.

    Args:
        cap:          OpenCV VideoCapture positioned at frame 0.
        ingest_data:  VideoIngest metadata (fps, roi, frame_count).
        output_dir:   Root output directory; keyframes/ sub-dir is created here.

    Returns:
        registry            — final ElementRegistry after all pen-lifts
        segments            — list[TemporalSegment] ordered by timestamp
        registry_snapshots  — list[ElementRegistry] at each pen-lift (parallel to segments)
        keyframe_annotations — list[KeyframeAnnotation] at each pen-lift (parallel to segments)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rx, ry, rw, rh = ingest_data.roi

    gmm = cv2.createBackgroundSubtractorMOG2(
        history=config.GMM_HISTORY,
        varThreshold=config.GMM_VAR_THRESHOLD,
        detectShadows=False,
    )

    registry = ElementRegistry()
    segments: list[TemporalSegment] = []
    registry_snapshots: list[ElementRegistry] = []
    keyframe_annotations: list[KeyframeAnnotation] = []

    is_active = False
    idle_count = 0
    keyframe_before: np.ndarray | None = None
    segment_start_ts = 0.0
    segment_id = 0

    prev_gray: np.ndarray | None = None
    delta_magnitude = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    total = ingest_data.frame_count

    log.info("Stage 1: processing %d frames...", total)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[ry : ry + rh, rx : rx + rw]
        timestamp = frame_idx / ingest_data.fps

        # GMM foreground mask 
        fg_mask = gmm.apply(roi_frame)
        fg_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size

        # SSIM (sampled every N frames) 
        if frame_idx % config.SSIM_SAMPLE_INTERVAL == 0:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                score = ssim(prev_gray, gray, data_range=255)
                delta_magnitude = float(1.0 - score)
            prev_gray = gray

        # Hysteresis temporal state machine 
        if not is_active:
            if fg_ratio > config.ACTIVITY_THRESHOLD_HIGH:
                is_active = True
                idle_count = 0
                keyframe_before = roi_frame.copy()
                segment_start_ts = timestamp
                log.debug("Activity started at %.2fs (fg=%.4f)", timestamp, fg_ratio)
        else:
            if fg_ratio < config.ACTIVITY_THRESHOLD_LOW:
                idle_count += 1
                if idle_count >= config.STABILITY_WINDOW:
                    is_active = False
                    idle_count = 0
                    keyframe_after = roi_frame.copy()

                    # Detect elements on the stable post-activity frame
                    _, binary, edges = _preprocess(roi_frame)
                    gray_stable = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    nodes = _detect_nodes(binary, gray_stable)
                    node_bboxes = [d["bbox"] for d in nodes]
                    connections = _detect_connections(edges, node_bboxes)

                    # Cap total marks before registry update
                    all_detections = _apply_marks_cap(nodes + connections)
                    registry.update(
                        all_detections,
                        timestamp,
                        match_threshold=config.CENTROID_MATCH_THRESHOLD,
                    )

                    seg_type = _classify_segment_type(keyframe_before, keyframe_after)

                    seg = TemporalSegment(
                        segment_id=segment_id,
                        timestamp_start=segment_start_ts,
                        timestamp_end=timestamp,
                        segment_type=seg_type,
                        delta_magnitude=delta_magnitude,
                        keyframe_before=keyframe_before,
                        keyframe_after=keyframe_after,
                    )
                    segments.append(seg)
                    registry_snapshots.append(_snapshot_registry(registry))

                    # Write annotated keyframe PNG for VLM snapshot analysis
                    kf = _write_keyframe(keyframe_after, registry, segment_id, timestamp, output_dir)
                    keyframe_annotations.append(kf)

                    segment_id += 1
                    prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

                    log.debug(
                        "Pen-lift at %.2fs: segment %d type=%s marks=%d",
                        timestamp, segment_id - 1, seg_type, len(registry.elements),
                    )
            else:
                idle_count = 0

        if frame_idx % 300 == 0:
            log.info("  frame %d / %d  (%.1f%%)", frame_idx, total, 100 * frame_idx / max(total, 1))

        frame_idx += 1

    log.info(
        "Stage 1 done: %d segments, %d active marks, %d keyframes written",
        len(segments), len(registry.elements), len(keyframe_annotations),
    )
    return registry, segments, registry_snapshots, keyframe_annotations
