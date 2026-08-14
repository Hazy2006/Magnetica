import numpy as np

class Particle:
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)   # where it is
        self.velocity = np.array(velocity, dtype=float)   # how it's moving
        self.charge = charge                              # q
        self.mass = mass                                  # m

    def step(self, bz, dt):
        vx, vy = self.velocity

        # 1. Lorentz force (2D)
        fx = self.charge * (vy * bz)
        fy = self.charge * (-vx * bz)

        # 2. force -> acceleration
        ax = fx / self.mass
        ay = fy / self.mass

        # 3. acceleration -> velocity
        self.velocity += np.array([ax, ay]) * dt

        # 4. velocity -> position
        self.position += self.velocity * dt

    def step_attracted(self, magnets, dt, g=1.0):
        """Steps the particle under a pull toward every magnet, strength
        scaling with the magnet's moment magnitude and falling off with
        the square of distance (softened near the core, Plummer-style,
        so it never singularizes on close approach). This isn't real
        magnetism -- the Lorentz force in `step` can't produce closed
        orbits (it does no work, so it never pulls anything back in).
        This is a stand-in central force chosen so the particle traces a
        bounded path shaped by the magnets' layout instead of scattering
        off to infinity."""
        force = np.zeros(2)
        for magnet in magnets:
            offset = magnet.position - self.position
            strength = np.linalg.norm(magnet.moment)
            r_sq = offset[0] ** 2 + offset[1] ** 2
            softened = (r_sq + magnet.radius ** 2) ** 1.5
            force += g * strength * offset / softened

        self.velocity += (force / self.mass) * dt
        self.position += self.velocity * dt

    def clamp_to_bounds(self, xlim, ylim):
        """Bounces the particle elastically off a rectangular box. A close
        pass by a magnet can inject enough numerical energy in one step to
        send the particle to infinity (see step_attracted); rather than
        chase that with a smaller timestep, treat the plotted area as a
        wall so the particle stays visible instead of escaping the frame."""
        for axis, (lo, hi) in enumerate((xlim, ylim)):
            if self.position[axis] < lo:
                self.position[axis] = lo
                self.velocity[axis] *= -1
            elif self.position[axis] > hi:
                self.position[axis] = hi
                self.velocity[axis] *= -1