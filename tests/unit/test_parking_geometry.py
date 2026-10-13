"""Unit tests for planar geometry, polygon validation, and coordinate scaling."""

import pytest
from app.parking.geometry import (
    validate_slot_polygon,
    calculate_overlap_ratio,
    check_slot_overlaps,
    denormalize_points,
    normalize_points,
)


def test_valid_polygon():
    pts = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]
    valid, err = validate_slot_polygon(pts)
    assert valid is True
    assert err is None


def test_self_intersecting_polygon_rejected():
    # Bow-tie / figure-eight polygon
    bowtie = [[0.1, 0.1], [0.3, 0.4], [0.3, 0.1], [0.1, 0.4]]
    valid, err = validate_slot_polygon(bowtie)
    assert valid is False
    assert "self-intersecting" in err.lower() or "invalid" in err.lower()


def test_tiny_subscale_polygon_rejected():
    # Area = 0.001 * 0.001 = 1e-6 (far below min_area 0.005)
    tiny = [[0.100, 0.100], [0.101, 0.100], [0.101, 0.101], [0.100, 0.101]]
    valid, err = validate_slot_polygon(tiny, min_area=0.005)
    assert valid is False
    assert "smaller than minimum threshold" in err.lower()


def test_out_of_bounds_polygon_rejected():
    out_of_bounds = [[0.1, 0.1], [1.2, 0.1], [0.3, 0.4], [0.1, 0.4]]
    valid, err = validate_slot_polygon(out_of_bounds)
    assert valid is False
    assert "outside normalized" in err.lower()


def test_overlap_ratio_calculation():
    # Slot polygon: [0.2, 0.2] to [0.4, 0.4] -> Area = 0.04
    slot_pts = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]

    # Case A: Fully covering bounding box
    bbox_full = (0.1, 0.1, 0.5, 0.5)
    ratio_full = calculate_overlap_ratio(slot_pts, bbox_full)
    assert pytest.approx(ratio_full, 0.01) == 1.0

    # Case B: Disjoint / non-overlapping box
    bbox_disjoint = (0.6, 0.6, 0.8, 0.8)
    ratio_disjoint = calculate_overlap_ratio(slot_pts, bbox_disjoint)
    assert ratio_disjoint == 0.0

    # Case C: 50% horizontal overlap (xmin: 0.2 to 0.3)
    bbox_half = (0.2, 0.2, 0.3, 0.4)
    ratio_half = calculate_overlap_ratio(slot_pts, bbox_half)
    assert pytest.approx(ratio_half, 0.02) == 0.50


def test_slot_overlap_warnings():
    slot_a = {"id": "S01", "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]}
    slot_b_non_overlapping = {"id": "S02", "points": [[0.5, 0.1], [0.7, 0.1], [0.7, 0.4], [0.5, 0.4]]}
    slot_c_overlapping = {"id": "S03", "points": [[0.2, 0.1], [0.4, 0.1], [0.4, 0.4], [0.2, 0.4]]}

    assert len(check_slot_overlaps([slot_a, slot_b_non_overlapping])) == 0

    warnings = check_slot_overlaps([slot_a, slot_c_overlapping], max_allowed_overlap=0.10)
    assert len(warnings) == 1
    assert "overlap" in warnings[0].lower()


def test_coordinate_denormalization():
    norm_pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    pixels = denormalize_points(norm_pts, width=1280, height=720)
    assert pixels == [(0, 0), (640, 360), (1280, 720)]


def test_is_point_in_slot():
    from app.parking.geometry import is_point_in_slot
    slot = [[0.2, 0.2], [0.5, 0.2], [0.5, 0.6], [0.2, 0.6]]
    assert is_point_in_slot(slot, (0.35, 0.4)) is True
    assert is_point_in_slot(slot, (0.1, 0.1)) is False
    assert is_point_in_slot(slot, (0.6, 0.5)) is False


def test_hybrid_spatial_association_score():
    from app.parking.geometry import calculate_spatial_association_score
    slot = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.6], [0.2, 0.6]]

    # Case 1: Vehicle parked neatly inside bay
    # bbox: [0.22, 0.25, 0.38, 0.55] -> bottom center = (0.30, 0.55), inside slot
    bbox_inside = (0.22, 0.25, 0.38, 0.55)
    score, reason = calculate_spatial_association_score(slot, bbox_inside, method="hybrid")
    assert score > 0.5
    assert "inside bay" in reason.lower()

    # Case 2: Vehicle parked in adjacent bay whose roof overhangs into this slot
    # bbox: [0.35, 0.2, 0.55, 0.55] -> bottom center = (0.45, 0.55), which is outside slot!
    bbox_overhang = (0.35, 0.2, 0.55, 0.55)
    score_overhang, reason_overhang = calculate_spatial_association_score(slot, bbox_overhang, method="hybrid")
    # Dampened score because wheels are outside the bay
    assert score_overhang < 0.35
    assert "outside bay" in reason_overhang.lower()


def test_automatic_slot_discovery():
    from app.parking.geometry import discover_candidate_slots_from_detections
    bboxes = [
        (0.10, 0.20, 0.25, 0.50),
        (0.30, 0.20, 0.45, 0.50),
        (0.50, 0.20, 0.65, 0.50),
    ]
    discovered = discover_candidate_slots_from_detections(bboxes)
    assert len(discovered) == 3
    assert discovered[0]["id"] == "AUTO-01"
    assert discovered[1]["id"] == "AUTO-02"
    assert discovered[2]["id"] == "AUTO-03"
    assert len(discovered[0]["points"]) == 4

