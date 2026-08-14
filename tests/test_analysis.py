import pytest
import numpy as np

from magnetica.analysis import OrbitAnalyzer
from magnetica.field import Magnet


def test_status_is_bounded_for_slow_nearby_particle():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    assert analyzer.status == "Bounded"


def test_status_is_unbound_for_fast_far_particle():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(10.0, 0.0), magnets=magnets)
    assert analyzer.status == "Unbound"


def test_max_radius_tracks_running_maximum_from_centroid():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(1.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    analyzer.update(0.1, position=(3.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    analyzer.update(0.2, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    assert analyzer.max_radius == pytest.approx(3.0)


def test_properties_default_before_any_crossing():
    analyzer = OrbitAnalyzer(g=1.0)
    assert analyzer.period is None
    assert analyzer.closure_pct is None
    assert analyzer.is_convex is None
    assert analyzer.crossing_count == 0
    assert analyzer.poincare_points == []


def test_period_and_closure_for_exact_circular_motion():
    R = 2.0
    T = 4.0
    w = 2 * np.pi / T
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    # Start at angle = pi (well away from the crossing ray at angle 0)
    # and run for just over 2 full periods.
    steps = int(2.2 * T / dt)
    for i in range(steps):
        t = i * dt
        theta = np.pi + w * t
        position = (R * np.cos(theta), R * np.sin(theta))
        velocity = (-R * w * np.sin(theta), R * w * np.cos(theta))
        analyzer.update(t, position, velocity, magnets=[])

    assert analyzer.period is not None
    assert abs(analyzer.period - T) < 0.05
    assert analyzer.closure_pct is not None
    assert analyzer.closure_pct > 99.0
    assert len(analyzer.poincare_points) == 2


def test_no_crossing_within_debounce_interval():
    analyzer = OrbitAnalyzer(g=1.0)
    # Two samples that cross the ray twice within 0.1s of each other
    # (well under the 0.5s debounce) should only register once.
    analyzer.update(0.0, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.05, position=(1.0, 0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.1, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    assert len(analyzer.poincare_points) == 1


from magnetica.analysis import segments_intersect, analyze_lap


def test_segments_intersect_detects_crossing_x():
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0)) is True


def test_segments_intersect_ignores_parallel_segments():
    assert segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)) is False


def test_analyze_lap_convex_circle_has_no_self_intersections():
    n = 40
    points = [(np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)]
    is_convex, crossings = analyze_lap(points)
    assert is_convex is True
    assert crossings == 0


def test_analyze_lap_star_is_not_convex_and_self_intersects():
    # A pentagram traced through every second vertex of a regular pentagon.
    # Note: this turns the same rotational direction at every vertex (just
    # a wider angle than a simple pentagon), so a naive "same-sign turns"
    # check alone would wrongly call it convex -- it must also fail because
    # it self-intersects. That's exactly what this test guards.
    n = 5
    outer = [
        (np.cos(2 * np.pi * i / n - np.pi / 2), np.sin(2 * np.pi * i / n - np.pi / 2))
        for i in range(n)
    ]
    star_order = [0, 2, 4, 1, 3]
    points = [outer[i] for i in star_order]
    is_convex, crossings = analyze_lap(points)
    assert crossings > 0
    assert is_convex is False


def test_analyze_lap_returns_none_for_too_few_points():
    is_convex, crossings = analyze_lap([(0, 0), (1, 1)])
    assert is_convex is None
    assert crossings == 0


def test_energy_history_rolls_off_old_samples_outside_window():
    dt = 0.1
    window = 1.0
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt, energy_window_seconds=window)
    for i in range(30):
        t = i * dt
        analyzer.update(t, position=(1.0, 0.0), velocity=(0.0, 0.0), magnets=[])
    hist = analyzer.energy_history
    assert len(hist["t"]) == 10
    assert hist["t"][0] == pytest.approx(2.0)
    assert hist["t"][-1] == pytest.approx(2.9)


def test_relative_energy_error_matches_formula():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=0.1)

    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)
    e0 = analyzer.energy_history["e"][-1]

    analyzer.update(0.1, position=(2.0, 0.0), velocity=(0.0, 2.0), magnets=magnets)
    e1 = analyzer.energy_history["e"][-1]
    expected_rel_err = abs(e1 - e0) / abs(e0)

    assert analyzer.energy_history["rel_err"][-1] == pytest.approx(expected_rel_err)
