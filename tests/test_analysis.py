import pytest
import numpy as np

from magnetica.analysis import OrbitAnalyzer, LAP_POINT_CAP, MODE_ATTRACTION, MODE_LORENTZ, segments_intersect, analyze_lap
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


def test_lorentz_mode_analyzer_reports_no_status_or_potential_energy():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, mode=MODE_LORENTZ)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)

    assert analyzer.status is None
    hist = analyzer.energy_history
    assert hist["ke"] == [pytest.approx(0.5)]
    assert hist["pe"] == []
    assert hist["e"] == []
    assert hist["rel_err"] == []


def test_attraction_mode_analyzer_default_still_reports_status_and_pe():
    # Default mode is unchanged -- existing call sites like
    # OrbitAnalyzer(g=1.0) elsewhere in this file must keep working.
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)

    assert analyzer.status == "Bounded"
    hist = analyzer.energy_history
    assert len(hist["pe"]) == 1
    assert len(hist["e"]) == 1


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


def test_crossing_events_increments_every_crossing_unlike_crossing_count():
    # crossing_count (self-intersections of a *completed* lap) only updates
    # once every two crossings and stays 0 for a simple closed orbit, which
    # made the analyzer table's "Crossings" field look stuck at 0 even while
    # the Poincare/lap-radius plot kept gaining points every crossing.
    # crossing_events tracks the same thing the plot is fed from.
    R = 2.0
    T = 4.0
    w = 2 * np.pi / T
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    steps = int(2.2 * T / dt)
    for i in range(steps):
        t = i * dt
        theta = np.pi + w * t
        position = (R * np.cos(theta), R * np.sin(theta))
        velocity = (-R * w * np.sin(theta), R * w * np.cos(theta))
        analyzer.update(t, position, velocity, magnets=[])

    assert analyzer.crossing_events == len(analyzer.poincare_points) == 2
    assert analyzer.crossing_count == 0  # a circle never self-intersects

    lap_hist = analyzer.lap_radius_history
    assert len(lap_hist) == 2
    assert [lap for lap, _ in lap_hist] == [1, 2]
    assert lap_hist[0][1] == pytest.approx(R, abs=0.05)


def test_no_crossing_within_debounce_interval():
    analyzer = OrbitAnalyzer(g=1.0)
    # Two samples that cross the ray twice within 0.1s of each other
    # (well under the 0.5s debounce) should only register once.
    analyzer.update(0.0, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.05, position=(1.0, 0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.1, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    assert len(analyzer.poincare_points) == 1


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


def test_is_convex_and_crossing_count_stay_default_after_one_crossing():
    # Same exact-circular-motion setup as
    # test_period_and_closure_for_exact_circular_motion, but stopped after
    # only ~1.2 periods -- long enough for exactly one ray crossing (which
    # happens once per revolution, at t = T/2) but short enough to miss the
    # second one (at t = 3T/2). is_convex/crossing_count require a full
    # completed lap (2 crossings), so they must still read their Task-1
    # defaults here even though _lap_points has accumulated plenty of data.
    R = 2.0
    T = 4.0
    w = 2 * np.pi / T
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    steps = int(1.2 * T / dt)
    for i in range(steps):
        t = i * dt
        theta = np.pi + w * t
        position = (R * np.cos(theta), R * np.sin(theta))
        velocity = (-R * w * np.sin(theta), R * w * np.cos(theta))
        analyzer.update(t, position, velocity, magnets=[])

    # Sanity check: the test actually exercised exactly one crossing, not
    # zero (which would make the assertions below trivially true).
    assert len(analyzer.poincare_points) == 1

    assert analyzer.is_convex is None
    assert analyzer.crossing_count == 0


def test_lap_points_never_exceed_cap_with_no_crossings():
    # A position with rel[0] < 0 relative to the (empty-magnets) centroid
    # can never satisfy the crossing condition (sign_changed and
    # rel[0] > 0), so no crossing ever fires and _lap_points would grow
    # without bound if it weren't capped.
    analyzer = OrbitAnalyzer(g=1.0)
    for i in range(5000):
        t = i * 0.01
        analyzer.update(t, position=(-5.0, 0.0), velocity=(0.0, 0.0), magnets=[])

    assert len(analyzer._lap_points) <= LAP_POINT_CAP
    assert analyzer.crossing_count == 0


def test_relative_energy_error_matches_formula():
    # e0 here (~ke0 - |pe0|) lands close to zero, which is exactly the
    # near-zero-total-energy scenario that used to blow up the relative
    # error under abs(e0) normalization -- see the regression test below.
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=0.1)

    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)
    e0 = analyzer.energy_history["e"][-1]
    ke0 = analyzer.energy_history["ke"][-1]
    pe0 = analyzer.energy_history["pe"][-1]

    analyzer.update(0.1, position=(2.0, 0.0), velocity=(0.0, 2.0), magnets=magnets)
    e1 = analyzer.energy_history["e"][-1]
    expected_scale = abs(ke0) + abs(pe0)
    expected_rel_err = abs(e1 - e0) / expected_scale

    assert analyzer.energy_history["rel_err"][-1] == pytest.approx(expected_rel_err)


def test_relative_error_stays_bounded_when_total_energy_near_zero():
    # Same near-zero e0 as above. Under the old abs(e0)-normalized formula
    # this tiny denominator turned an ordinary energy change into a
    # multi-thousand-percent "relative error"; the energy-scale
    # normalization keeps it sane instead.
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=0.1)

    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)
    assert abs(analyzer.energy_history["e"][-1]) < 0.01  # sanity: e0 really is near zero

    analyzer.update(0.1, position=(2.0, 0.0), velocity=(0.0, 1.001), magnets=magnets)
    assert analyzer.energy_history["rel_err"][-1] < 1.0


def test_self_crossing_is_detected_even_when_it_straddles_a_lap_boundary():
    # Two overlapping circles that form a self-intersecting path when traced
    # in sequence. The overlap region represents a "self-crossing". By tracing
    # them in the right order, the crossing straddles a lap boundary (the 2nd
    # crossing), so the old reset logic would discard points before the
    # crossing is detected. With the new code, the rolling buffer preserves
    # both lobes and detects the crossing.
    R = 1.0
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    t = 0.0
    n = 200

    # First circle: centered at (1.5R, 0), radius R.
    # Crosses Poincare ray at x=2.5R and x=0.5R (both > 0).
    for i in range(n + 1):
        theta = 2 * np.pi * i / n
        position = (1.5 * R + R * np.cos(theta), R * np.sin(theta))
        velocity = (0.0, 0.0)
        analyzer.update(t, position, velocity, magnets=[])
        t += dt

    # Second circle: centered at (2.5R, 0), radius R.
    # Crosses Poincare ray at x=3.5R and x=1.5R (both > 0).
    # These two circles overlap, so when both are in _lap_points,
    # their intersection is detected as a self-crossing.
    for i in range(n + 1):
        theta = 2 * np.pi * i / n
        position = (2.5 * R + R * np.cos(theta), R * np.sin(theta))
        velocity = (0.0, 0.0)
        analyzer.update(t, position, velocity, magnets=[])
        t += dt

    assert analyzer.crossing_count > 0
