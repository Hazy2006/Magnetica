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
