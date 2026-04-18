from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class TemporalSegment:
    segment_id: int
    timestamp_start: float          # seconds
    timestamp_end: float
    segment_type: str               # "activity" | "idle" | "erasure"
    delta_magnitude: float          # SSIM-based change score
    keyframe_before: np.ndarray     # stable frame before activity
    keyframe_after: np.ndarray      # stable frame after activity

@dataclass
class DetectedRegion:
    mark_id: int                    # SoM label number
    bbox: tuple[int,int,int,int]    # x, y, w, h
    shape_type: str                 # rectangle, diamond, oval, circle, triangle, other
    centroid: tuple[int, int]       # x, y
    contour: np.ndarray
    first_seen: float               # timestamp when first detected (seconds)

@dataclass
class ElementRegistry:
    """Persistent map of mark_id → DetectedRegion across the video.
    Provides mark ID stability: same element keeps same ID across frames."""
    elements: dict[int, DetectedRegion] = field(default_factory=dict)
    next_id: int = 1

    def update(
        self,
        detections: list[dict],
        timestamp: float,
        match_threshold: int = 50,
    ) -> list[DetectedRegion]:
        """Match new detections to existing elements by centroid proximity.

        detections: list of dicts with keys: bbox, shape_type, contour, centroid
        New elements get fresh IDs. Unmatched existing elements are retired.
        Returns the list of DetectedRegion objects (with IDs resolved).
        """
        matched_existing: set[int] = set()
        result: list[DetectedRegion] = []

        for det in detections:
            cx, cy = det["centroid"]
            best_id: int | None = None
            best_dist = float("inf")

            for mark_id, region in self.elements.items():
                rx, ry = region.centroid
                dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                if dist < best_dist and dist < match_threshold:
                    best_dist = dist
                    best_id = mark_id

            if best_id is not None:
                matched_existing.add(best_id)
                region = DetectedRegion(
                    mark_id=best_id,
                    bbox=det["bbox"],
                    shape_type=det["shape_type"],
                    centroid=det["centroid"],
                    contour=det["contour"],
                    first_seen=self.elements[best_id].first_seen,
                )
                self.elements[best_id] = region
                result.append(region)
            else:
                region = DetectedRegion(
                    mark_id=self.next_id,
                    bbox=det["bbox"],
                    shape_type=det["shape_type"],
                    centroid=det["centroid"],
                    contour=det["contour"],
                    first_seen=timestamp,
                )
                result.append(region)
                self.next_id += 1

        # Retire elements not matched to any new detection
        for uid in set(self.elements.keys()) - matched_existing:
            del self.elements[uid]

        # Register newly created elements
        for region in result:
            if region.mark_id not in self.elements:
                self.elements[region.mark_id] = region

        return result

@dataclass
class VLMOperation:
    """One operation from VLM analysis of the marked video."""
    operation_id: int
    operation_type: str             # CREATION | ADDITION | HIGHLIGHTING | ERASURE | COMPLETE_ERASURE
    timestamp_start: str            # MM:SS
    timestamp_end: str              # MM:SS
    confidence: str                 # high | medium | low
    marks_involved: list[int]       # mark IDs created/modified/removed
    per_mark_descriptions: dict[int, dict]  # mark_id → {text, element_type, semantic_role}
    connections: list[dict]         # {from_mark, to_mark, direction, label, line_type}
    classification_reasoning: dict
    pedagogical_context: str
    physical_action: dict
    visual_evidence: dict

@dataclass
class KeyframeAnnotation:
    """A single annotated keyframe PNG emitted at each pen-lift event.

    Sent to the VLM as a separate image file alongside the clean original video, giving spatial grounding
    without degrading the video's visual fidelity.
    """
    segment_id: int
    timestamp: float            # seconds — pen-lift moment (segment.timestamp_end)
    image_path: Path            # written PNG path
    marks: list[dict]           # [{mark_id, shape_type, centroid, bbox}] at this moment


@dataclass
class BoardSnapshot:
    """VLM analysis of a single keyframe for static IR generation.

    Produced by stage2.analyse_snapshots() for each pen-lift keyframe.
    Contains all fields needed to populate static-schema.json directly.
    CV registry provides ground-truth geometry; this provides semantics.
    """
    segment_id: int
    timestamp: float                    # pen-lift timestamp (seconds)
    mark_descriptions: dict[int, dict]  # mark_id → {text, shape, element_type, semantic_role, visual}
    connections: list[dict]             # {from_mark, to_mark, direction, line_type, label}
    symbol_meanings: list[dict]         # [{shape, meaning}] — VLM-verified legend
    groupings: list[dict]               # [{label, parent, members: [mark_id]}]
    annotations: list[dict]             # [{mark_id, annotation_type, content}]
    cross_links: list[dict]             # [{source_flowchart, target_flowchart, source_element, target_element, label}]
    board_state: str
    confidence: str                     # high | medium | low
    visibility_issues: str | None


@dataclass
class GroundedElement:
    """A DetectedRegion enriched by VLM semantic analysis."""
    mark_id: int
    bbox: tuple[int,int,int,int]    # from CV (ElementRegistry)
    shape: str                      # from CV
    centroid: tuple[int, int]       # from CV
    text: str | None                # from VLM
    element_type: str               # node | connection | annotation (from VLM)
    connections_to: list[int] = field(default_factory=list)     # mark IDs (from VLM)
    vlm_description: str = ""           # from VLM