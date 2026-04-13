"""Stage 1: CV processing, SoM marking, and marked video generation.

Single pass through the ROI-cropped video:
  1. GMM background subtraction -> foreground ratio per frame
  2. SSIM between sampled frames -> perceptual change magnitude
  3. Hysteresis thresholding -> activity / idle temporal segmentation
  4. On pen-lift: contour detection + shape classification -> ElementRegistry update
  5. Every frame: overlay SoM marks (numbered labels + colored bboxes)
  6. Write marked frames to marked_video.mp4

Outputs:
  - ElementRegistry (mark_id -> DetectedRegion, persists across video)
  - list[TemporalSegment] (one per activity window)
  - Path to marked_video.mp4
"""
from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pipeline import config
from pipeline.models import DetectedRegion, ElementRegistry, TemporalSegment
from pipeline.stage0_ingest import VideoIngest

log = logging.getLogger(__name__)

_CONNECTION_SHAPES = {"arrow", "line", "connection"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a BGR ROI frame to gray, binary, and edge images.

    Returns (gray, binary, edges).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE for uneven whiteboard lighting (ported from flowchart-conversion/preprocess.py)
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


# ---------------------------------------------------------------------------
# Shape classification (ported from flowchart-conversion/node_detection.py)
# ---------------------------------------------------------------------------

def _classify_shape(contour: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Classify a closed contour as one of: rectangle, diamond, oval, circle, triangle, other."""
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
        # Diamond: 4 vertices but the contour only fills ~half the bounding box
        # because the corners of the bbox are empty.
        # Rectangle: fills most of the bbox.
        return "rectangle" if extent >= 0.65 else "diamond"

    if circularity >= 0.6 or n >= 7:
        if extent > 0.5:
            return "circle" if 0.85 <= aspect <= 1.15 else "oval"

    return "other"


# ---------------------------------------------------------------------------
# Element detection on a stable frame
# ---------------------------------------------------------------------------

def _detect_nodes(binary: np.ndarray, gray: np.ndarray) -> list[dict]:
    """Detect closed-shape nodes (rect, diamond, oval, circle, triangle) via contours."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    Lines whose both endpoints are near a node bbox are classified as
    'connection'. All others are discarded (likely noise or internal marks).
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

    detections: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip lines internal to a single node bbox
        if _point_in_any_bbox(x1, y1, node_bboxes) and _point_in_any_bbox(x2, y2, node_bboxes):
            # Both endpoints in the same bbox? Check if the same one
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


# ---------------------------------------------------------------------------
# SoM overlay
# ---------------------------------------------------------------------------

def _draw_som_marks(
    frame: np.ndarray,
    registry: ElementRegistry,
    is_active: bool,
) -> np.ndarray:
    """Overlay numbered SoM labels and bounding boxes on a frame.

    Nodes get green boxes. Connections get blue lines/boxes.
    Active periods get a subtle yellow tint to signal activity to the VLM.
    """
    out = frame.copy()

    if is_active:
        # Subtle yellow tint: blend with yellow
        tint = np.zeros_like(out)
        tint[:] = config.MARK_DELTA_HIGHLIGHT
        out = cv2.addWeighted(out, 0.85, tint, 0.15, 0)

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

        # Dark background for readability
        cv2.rectangle(out, (lx, ly - th - baseline), (lx + tw + 2, ly + baseline), (0, 0, 0), -1)
        cv2.putText(out, label, (lx + 1, ly), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Segment type classification
# ---------------------------------------------------------------------------

def _classify_segment_type(
    before: np.ndarray, after: np.ndarray
) -> str:
    """Classify an activity segment as 'activity' or 'erasure'.

    Erasure = ink density dropped significantly from before to after.
    Ink density = fraction of dark pixels (ink on whiteboard).
    """
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Dark pixels = ink
    density_before = np.mean(gray_before < 128)
    density_after = np.mean(gray_after < 128)

    if density_before > 0.01 and density_after < density_before * 0.70:
        return "erasure"
    return "activity"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    cap: cv2.VideoCapture,
    ingest_data: VideoIngest,
    output_dir: Path,
) -> tuple[ElementRegistry, list[TemporalSegment], Path]:
    """Process the full video: temporal segmentation + SoM marking.

    Returns:
        registry         — final ElementRegistry (mark_id -> DetectedRegion)
        segments         — list[TemporalSegment] ordered by timestamp
        marked_video_path — path to the output marked_video.mp4
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    marked_video_path = output_dir / "marked_video.mp4"

    rx, ry, rw, rh = ingest_data.roi
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(marked_video_path), fourcc, ingest_data.fps, (rw, rh))

    gmm = cv2.createBackgroundSubtractorMOG2(
        history=config.GMM_HISTORY,
        varThreshold=config.GMM_VAR_THRESHOLD,
        detectShadows=False,
    )

    registry = ElementRegistry()
    segments: list[TemporalSegment] = []

    # Temporal state machine
    is_active = False
    idle_count = 0
    keyframe_before: np.ndarray | None = None
    segment_start_ts = 0.0
    segment_id = 0

    # SSIM tracking
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

        # --- GMM foreground mask ---
        fg_mask = gmm.apply(roi_frame)
        fg_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size

        # --- SSIM (every N frames) ---
        if frame_idx % config.SSIM_SAMPLE_INTERVAL == 0:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                score = ssim(prev_gray, gray, data_range=255)
                delta_magnitude = float(1.0 - score)
            prev_gray = gray

        # --- Hysteresis temporal state machine ---
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
                    # Pen lift: transition back to idle
                    is_active = False
                    idle_count = 0
                    keyframe_after = roi_frame.copy()

                    # Update registry on the stable post-activity frame
                    _, binary, edges = _preprocess(roi_frame)
                    gray_stable = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    nodes = _detect_nodes(binary, gray_stable)
                    node_bboxes = [d["bbox"] for d in nodes]
                    connections = _detect_connections(edges, node_bboxes)
                    registry.update(
                        nodes + connections,
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
                    segment_id += 1

                    log.debug(
                        "Pen-lift at %.2fs: segment %d type=%s marks=%d",
                        timestamp, segment_id - 1, seg_type, len(registry.elements),
                    )

                    # Update prev_gray for SSIM continuity
                    prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                idle_count = 0

        # --- SoM overlay ---
        marked = _draw_som_marks(roi_frame, registry, is_active)
        writer.write(marked)

        if frame_idx % 300 == 0:
            log.info("  frame %d / %d  (%.1f%%)", frame_idx, total, 100 * frame_idx / max(total, 1))

        frame_idx += 1

    writer.release()
    log.info(
        "Stage 1 done: %d segments, %d active marks, marked video -> %s",
        len(segments), len(registry.elements), marked_video_path,
    )
    return registry, segments, marked_video_path
