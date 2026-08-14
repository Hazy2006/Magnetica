"""
Particle trajectory test — the scenario previously run as a manual
diagnostic (magnetica/main.py's print-and-plot loop) turned into an
assertion.

The Lorentz magnetic force is always perpendicular to velocity, so it
does no work: speed should stay close to its initial value as the
particle orbits. If integration is unstable, speed grows without bound.
"""
import numpy as np

from magnetica.field import Magnet, compute_bz
from magnetica.particle import Particle


def test_particle_speed_stays_bounded_in_orbit():
    magnets = [Magnet(position=(0, 0), moment=(0, 2), radius=0.1)]
    p = Particle(position=(1, 0), velocity=(0, 2), charge=1.0)
    dt = 0.001
    steps = 3200

    initial_speed = np.linalg.norm(p.velocity)
    max_speed = initial_speed

    for _ in range(steps):
        px, py = p.position
        bz = compute_bz(magnets, np.array([px]), np.array([py]))[0]
        p.step(bz, dt)
        max_speed = max(max_speed, np.linalg.norm(p.velocity))

    assert max_speed < initial_speed * 10


def test_particle_clamps_to_bounds_and_reflects_velocity():
    xlim, ylim = (-5, 5), (-5, 5)
    p = Particle(position=(4.99, 0), velocity=(10, 0), charge=1.0)

    # A step big enough to overshoot the right wall in one go.
    p.position += p.velocity * 0.1
    p.clamp_to_bounds(xlim, ylim)

    assert p.position[0] == 5
    assert p.velocity[0] == -10


def test_particle_stays_in_bounds_through_close_magnet_encounter():
    """Reproduces the slingshot: a close pass by a magnet injects enough
    numerical energy to fling the particle onto an escaping trajectory.
    Clamping every substep should keep it visible instead of losing it
    off the edge of the plot."""
    xlim, ylim = (-5, 5), (-5, 5)
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    p = Particle(position=(0.5, 2.0), velocity=(0.0, -1.0), charge=1.0)
    g = 0.5
    dt = 0.01

    for _ in range(2000):
        p.step_attracted(magnets, dt, g=g)
        p.clamp_to_bounds(xlim, ylim)
        assert xlim[0] <= p.position[0] <= xlim[1]
        assert ylim[0] <= p.position[1] <= ylim[1]
