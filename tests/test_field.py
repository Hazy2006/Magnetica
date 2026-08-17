import pytest
import numpy as np

from magnetica.field import Magnet, compute_attraction_field


def test_attraction_field_points_toward_single_magnet_and_matches_acceleration_formula():
    magnet = Magnet(position=(0, 0), moment=(0, 2), radius=0.1)
    g = 0.5
    x = np.array([[2.0]])
    y = np.array([[0.0]])

    fx, fy = compute_attraction_field([magnet], x, y, g)

    offset = magnet.position - np.array([2.0, 0.0])
    strength = np.linalg.norm(magnet.moment)
    r_sq = offset[0] ** 2 + offset[1] ** 2
    softened = (r_sq + magnet.radius ** 2) ** 1.5
    expected = g * strength * offset / softened

    assert fx[0, 0] == pytest.approx(expected[0])
    assert fy[0, 0] == pytest.approx(expected[1])
    assert fx[0, 0] < 0  # magnet is to the left, so the pull points in -x
