import numpy as np

class Particle:
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge = charge
        self.mass = mass

    def step(self, bz, dt):
        """Rotates velocity by the exact gyration angle for one step (bz is
        purely out-of-plane, so the true motion is a rotation) -- the
        out-of-plane-B specialization of the Boris pusher. Preserves |v|
        exactly, unlike the semi-implicit Euler update this replaced."""
        theta = -(self.charge * bz / self.mass) * dt
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        vx, vy = self.velocity
        self.velocity = np.array([vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t])
        self.position += self.velocity * dt

    def _attraction_acceleration(self, magnets, g):
        force = np.zeros(2)
        for magnet in magnets:
            offset = magnet.position - self.position
            strength = np.linalg.norm(magnet.moment)
            r_sq = offset[0] ** 2 + offset[1] ** 2
            softened = (r_sq + magnet.radius ** 2) ** 1.5
            force += g * strength * offset / softened
        return force / self.mass

    def step_attracted(self, magnets, dt, g=1.0):
        """Steps the particle under a Plummer-softened central pull toward
        every magnet (not real magnetism -- a stand-in force so the
        particle orbits instead of scattering to infinity). Integrated
        with velocity Verlet (symplectic, 2nd order) to keep energy drift
        small and bounded rather than growing over time."""
        a0 = self._attraction_acceleration(magnets, g)
        self.position += self.velocity * dt + 0.5 * a0 * dt ** 2
        a1 = self._attraction_acceleration(magnets, g)
        self.velocity += 0.5 * (a0 + a1) * dt

    def clamp_to_bounds(self, xlim, ylim):
        """Bounces the particle elastically off a rectangular box."""
        for axis, (lo, hi) in enumerate((xlim, ylim)):
            if self.position[axis] < lo:
                self.position[axis] = lo
                self.velocity[axis] *= -1
            elif self.position[axis] > hi:
                self.position[axis] = hi
                self.velocity[axis] *= -1