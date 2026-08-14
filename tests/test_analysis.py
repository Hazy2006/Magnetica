import pytest

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
