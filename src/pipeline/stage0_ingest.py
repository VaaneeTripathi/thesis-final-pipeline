"""Stage 0: Video ingest and ROI extraction.

Loads the lecture video, extracts metadata (fps, duration, resolution),
and detects the whiteboard region of interest (ROI) from the median of
the first 30 seconds. All downstream stages operate on ROI-cropped frames.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline import config

log = logging.getLogger(__name__)


@dataclass
class VideoIngest:
    fps: float
    frame_count: int
    duration: float                  # seconds
    width: int                       # full frame width (pre-crop)
    height: int                      # full frame height (pre-crop)
    roi: tuple[int, int, int, int]   # x, y, w, h of whiteboard in full frame


def ingest(video_path: Path) -> tuple[cv2.VideoCapture, VideoIngest]:
    """Load video and detect the whiteboard ROI.

    Returns an open VideoCapture (caller must release) and VideoIngest metadata.
    The capture is rewound to frame 0 before returning.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps

    log.info(
        "Video: %s  %.1fs  %dx%d  %.2f fps  %d frames",
        video_path.name, duration, width, height, fps, frame_count,
    )

    if config.ROI_MODE == "manual" and config.ROI_MANUAL is not None:
        roi = config.ROI_MANUAL
        log.info("ROI (manual): x=%d y=%d w=%d h=%d", *roi)
    else:
        roi = _detect_roi(cap, width, height, fps)
        log.info("ROI (auto):   x=%d y=%d w=%d h=%d", *roi)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, VideoIngest(
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        width=width,
        height=height,
        roi=roi,
    )


def _detect_roi(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: float,
) -> tuple[int, int, int, int]:
    """Detect whiteboard ROI from median brightness of the first 30 seconds.

    Strategy: compute per-pixel median across ~30 sampled frames, then find
    the largest bright rectangular region (the whiteboard surface).
    Falls back to the full frame if no suitable region is found.
    """
    sample_limit = int(min(30.0 * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    step = max(1, sample_limit // 30)

    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(0, sample_limit, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not frames:
        log.warning("ROI detection: no frames sampled — using full frame")
        return (0, 0, width, height)

    median_frame = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    # Whiteboard is bright; threshold to isolate it
    _, bright = cv2.threshold(median_frame, 200, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("ROI detection: no bright region found — using full frame")
        return (0, 0, width, height)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    coverage = (w * h) / (width * height)
    if coverage < 0.20:
        log.warning(
            "ROI detection: largest bright region is only %.1f%% of frame — using full frame",
            coverage * 100,
        )
        return (0, 0, width, height)

    return (x, y, w, h)
