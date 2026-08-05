"""
Charged particle and its motion through the field.

Day 3 lives here: a particle has a charge, a position, a velocity.
Each timestep, compute the Lorentz force from the field, use it to
update velocity, use velocity to update position. The trail of
positions is the path you see.

TODO (Day 3):
  - Particle state: charge, position, velocity.
  - Lorentz force:  F = q * (v x B).
  - One integration step (Euler first, then Runge-Kutta if needed).
"""

# TODO Day 3: implement particle + integration