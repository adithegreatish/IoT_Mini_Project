"""Planar geometry utilities, polygon validation, and coordinate scaling using Shapely."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, box


def normalize_points(points: List[List[float]], width: int, height: int) -> List[List[float]]:
    """Normalizes pixel coordinates to [0.0, 1.0] relative to reference width/height."""
    return [
        [round(max(0.0, min(1.0, p[0] / width)), 4), round(max(0.0, min(1.0, p[1] / height)), 4)]
        for p in points
    ]


def denormalize_points(points: List[List[float]], width: int, height: int) -> List[Tuple[int, int]]:
    """Converts normalized [0.0, 1.0] coordinates into integer pixel coordinates."""
    return [
        (int(round(p[0] * width)), int(round(p[1] * height)))
        for p in points
    ]


def validate_slot_polygon(points: List[List[float]], min_area: float = 0.0005) -> Tuple[bool, Optional[str]]:
    """
    Validates a slot polygon for:
    - Minimum vertex count (>= 3)
    - Normalized coordinate boundaries [0, 1] with floating point epsilon tolerance
    - Self-intersection / bow-ties
    - Minimum polygon area
    """
    if len(points) < 3:
        return False, "Polygon must contain at least 3 vertices."

    eps = 1e-4
    for idx, p in enumerate(points):
        if len(p) != 2:
            return False, f"Vertex {idx} does not have exactly (x, y) coordinates."
        if not (-eps <= p[0] <= (1.0 + eps) and -eps <= p[1] <= (1.0 + eps)):
            return False, f"Vertex {idx} coordinates ({p[0]}, {p[1]}) are outside normalized [0.0, 1.0] range."

    try:
        # Clamp vertices slightly outside [0, 1] due to precision
        clamped_points = [
            [max(0.0, min(1.0, float(pt[0]))), max(0.0, min(1.0, float(pt[1])))]
            for pt in points
        ]
        poly = Polygon(clamped_points)
        if not poly.is_valid:
            return False, "Polygon is self-intersecting or invalid."
        if poly.area < min_area:
            return False, f"Polygon normalized area {poly.area:.4f} is smaller than minimum threshold {min_area:.4f}."
    except Exception as e:
        return False, f"Geometry parsing error: {str(e)}"

    return True, None


def calculate_overlap_ratio(slot_points: List[List[float]], bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculates the intersection area ratio between a normalized bounding box [xmin, ymin, xmax, ymax]
    and a normalized slot polygon. Returns: Area(Intersection) / Area(Slot).
    """
    try:
        slot_poly = Polygon(slot_points)
        if not slot_poly.is_valid or slot_poly.area <= 0:
            return 0.0

        xmin, ymin, xmax, ymax = bbox
        veh_box = box(xmin, ymin, xmax, ymax)

        if not slot_poly.intersects(veh_box):
            return 0.0

        intersection = slot_poly.intersection(veh_box)
        return float(intersection.area / slot_poly.area)
    except Exception:
        return 0.0


def check_slot_overlaps(slots: List[Dict[str, Any]], max_allowed_overlap: float = 0.20) -> List[str]:
    """
    Checks if any two calibrated slot polygons overlap excessively.
    Returns list of warning descriptions.
    """
    warnings = []
    polys = []
    for s in slots:
        pts = s.get("points", [])
        if len(pts) >= 3:
            p = Polygon(pts)
            if p.is_valid:
                polys.append((s.get("id", "Unknown"), p))

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            id1, p1 = polys[i]
            id2, p2 = polys[j]
            if p1.intersects(p2):
                inter = p1.intersection(p2)
                overlap_ratio = inter.area / min(p1.area, p2.area)
                if overlap_ratio > max_allowed_overlap:
                    warnings.append(
                        f"Bays {id1} and {id2} overlap by {overlap_ratio * 100:.1f}% (threshold {max_allowed_overlap * 100:.1f}%)"
                    )
    return warnings


def is_point_in_slot(slot_points: List[List[float]], point: Tuple[float, float], boundary_tolerance: float = 0.01) -> bool:
    """
    Tests whether a 2D point (e.g. vehicle tire ground-contact point) lies inside or on the boundary of the slot polygon.
    Point coordinates must be normalized [0.0, 1.0]. Includes a boundary tolerance (default 1.0% of frame)
    to properly handle tires touching bay borders.
    """
    from shapely.geometry import Point
    try:
        poly = Polygon(slot_points)
        if not poly.is_valid or poly.area <= 0:
            return False
        pt = Point(point[0], point[1])
        return poly.covers(pt) or poly.distance(pt) <= boundary_tolerance
    except Exception:
        return False


def calculate_spatial_association_score(
    slot_points: List[List[float]],
    bbox: Tuple[float, float, float, float],
    method: str = "hybrid",
) -> Tuple[float, str]:
    """
    Computes perspective-aware vehicle-to-slot spatial association score.
    
    Handles camera perspective distortions:
    In elevated CCTV, a tall vehicle's roof projects into adjacent bays, while
    its wheels remain on the ground inside its designated bay.
    
    Methods:
    - 'overlap': standard polygon intersection ratio Area(Inter)/Area(Slot).
    - 'bottom_center': checks if the ground-contact point ((xmin+xmax)/2, ymax) is inside slot.
    - 'hybrid': combines overlap ratio with bottom-center containment.
      If bottom-center is inside, boosts confidence.
      If bottom-center is outside, dampens overlap to prevent adjacent bay stealing.
    
    Returns: (association_score: float in [0.0, 1.0], reason: str)
    """
    try:
        slot_poly = Polygon(slot_points)
        if not slot_poly.is_valid or slot_poly.area <= 0:
            return 0.0, "Invalid slot polygon"

        xmin, ymin, xmax, ymax = bbox
        veh_box = box(xmin, ymin, xmax, ymax)

        if not slot_poly.intersects(veh_box):
            return 0.0, "No geometric intersection"

        inter_area = slot_poly.intersection(veh_box).area
        overlap_ratio = float(inter_area / slot_poly.area)

        # Ground-contact bottom-center point: (center_x, bottom_y)
        bc_x = (xmin + xmax) / 2.0
        bc_y = ymax
        from shapely.geometry import Point
        pt = Point(bc_x, bc_y)
        bc_inside = slot_poly.covers(pt) or slot_poly.distance(pt) <= 0.01

        if method == "overlap":
            return round(overlap_ratio, 4), f"Area overlap: {overlap_ratio:.2f}"
        elif method == "bottom_center":
            score = 1.0 if bc_inside else 0.0
            return score, f"Bottom-center inside: {bc_inside}"
        else:  # hybrid
            if bc_inside:
                # Vehicle ground contact is inside slot; strong association
                score = min(1.0, max(0.5, overlap_ratio * 1.25))
                return round(score, 4), f"Ground contact inside bay (boosted overlap {overlap_ratio:.2f})"
            else:
                # Ground contact is outside bay; penalize apparent roof projection
                score = overlap_ratio * 0.65
                return round(score, 4), f"Ground contact outside bay (dampened overlap {overlap_ratio:.2f})"
    except Exception as e:
        return 0.0, f"Error: {e}"


def discover_candidate_slots_from_detections(
    bboxes: List[Tuple[float, float, float, float]],
    margin: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Mode B: Heuristic candidate parking-slot discovery from vehicle detections.
    Clusters parked vehicles into rows and extrapolates slot polygons.
    Manual calibration remains the primary authoritative fallback.
    """
    if not bboxes:
        return []

    # Sort boxes from left to right by center x
    sorted_boxes = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)
    discovered = []

    for idx, (xmin, ymin, xmax, ymax) in enumerate(sorted_boxes):
        # Expand slightly with margin to cover typical slot boundaries
        x1 = max(0.0, xmin - margin)
        y1 = max(0.0, ymin - margin)
        x2 = min(1.0, xmax + margin)
        y2 = min(1.0, ymax + margin)

        discovered.append({
            "id": f"AUTO-{idx+1:02d}",
            "label": f"Discovered Bay {idx+1}",
            "zone": "Auto-Zone",
            "level": "Ground",
            "slot_type": "car",
            "enabled": True,
            "mode": "automatic_discovery",
            "points": [
                [round(x1, 4), round(y1, 4)],
                [round(x2, 4), round(y1, 4)],
                [round(x2, 4), round(y2, 4)],
                [round(x1, 4), round(y2, 4)],
            ],
        })

    return discovered
