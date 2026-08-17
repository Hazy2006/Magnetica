"""
Particle trajectory test — the scenario previously run as a manual
diagnostic (magnetica/main.py's print-and-plot loop) turned into an
assertion.

The Lorentz magnetic force is always perpendicular to velocity, so it
does no work: speed should stay close to its initial value as the
particle orbits. If integration is unstable, speed grows without bound.
"""
import numpy as np
import pytest

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


def test_particle_speed_is_conserved_by_rotation_step():
    """The Lorentz force is always perpendicular to velocity and does no
    work, so |v| should be exactly conserved (to float precision) by a
    correct integrator -- not just bounded, as the loose 10x check above
    verifies. This is the test that would have caught the pre-Boris
    energy leak (~1.0015x per step, compounding to 2x -> 348x)."""
    magnets = [Magnet(position=(0, 0), moment=(0, 2), radius=0.1)]
    p = Particle(position=(1, 0), velocity=(0, 2), charge=1.0)
    dt = 0.001
    initial_speed = np.linalg.norm(p.velocity)

    for _ in range(3200):
        px, py = p.position
        bz = compute_bz(magnets, np.array([px]), np.array([py]))[0]
        p.step(bz, dt)
        assert np.linalg.norm(p.velocity) == pytest.approx(initial_speed, rel=1e-9)


def test_particle_clamps_to_bounds_and_reflects_velocity():
    xlim, ylim = (-5, 5), (-5, 5)
    p = Particle(position=(4.99, 0), velocity=(10, 0), charge=1.0)

    # A step big enough to overshoot the right wall in one go.
    p.position += p.velocity * 0.1
    p.clamp_to_bounds(xlim, ylim)

    assert p.position[0] == 5
    assert p.velocity[0] == -10


def test_step_attracted_conserves_energy_better_than_semi_implicit_euler():
    """step_attracted uses velocity Verlet (symplectic, 2nd order). Compare
    its energy drift on a close-encounter/wall-bounce scenario against a
    semi-implicit Euler step computed inline here (the integrator
    step_attracted used to use) -- Verlet should drift substantially less,
    since the old integrator is exactly what let the on-screen relative
    energy error spike so high."""
    magnet = Magnet(position=(0, 0), moment=(0, 3), radius=0.1)
    magnets = [magnet]
    g = 0.5
    xlim, ylim = (-5, 5), (-5, 5)
    dt = 0.01
    steps = 2000

    def energy(position, velocity):
        ke = 0.5 * float(np.dot(velocity, velocity))
        r = float(np.linalg.norm(magnet.position - position))
        strength = float(np.linalg.norm(magnet.moment))
        return ke - g * strength / np.sqrt(r ** 2 + magnet.radius ** 2)

    def euler_accel(position):
        offset = magnet.position - position
        strength = np.linalg.norm(magnet.moment)
        r_sq = offset[0] ** 2 + offset[1] ** 2
        softened = (r_sq + magnet.radius ** 2) ** 1.5
        return g * strength * offset / softened

    # Baseline: semi-implicit Euler, hand-rolled to match the old implementation.
    pos = np.array([0.5, 2.0])
    vel = np.array([0.0, -1.0])
    e0 = energy(pos, vel)
    euler_max_drift = 0.0
    for _ in range(steps):
        vel = vel + euler_accel(pos) * dt
        pos = pos + vel * dt
        for axis, (lo, hi) in enumerate((xlim, ylim)):
            if pos[axis] < lo:
                pos[axis], vel[axis] = lo, -vel[axis]
            elif pos[axis] > hi:
                pos[axis], vel[axis] = hi, -vel[axis]
        euler_max_drift = max(euler_max_drift, abs(energy(pos, vel) - e0) / abs(e0))

    # Same scenario through the real (Verlet) integrator.
    p = Particle(position=(0.5, 2.0), velocity=(0.0, -1.0), charge=1.0)
    e0 = energy(p.position, p.velocity)
    verlet_max_drift = 0.0
    for _ in range(steps):
        p.step_attracted(magnets, dt, g=g)
        p.clamp_to_bounds(xlim, ylim)
        verlet_max_drift = max(verlet_max_drift, abs(energy(p.position, p.velocity) - e0) / abs(e0))

    assert verlet_max_drift < euler_max_drift * 0.5


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
