"""
Magnetic field computation.

Day 1 lives here: define magnet sources, compute the field vector B
at any point in space. Everything downstream (rendering, particle
motion) reads from this.

TODO (Day 1):
  - Represent a magnet source (position, strength).
  - Compute the field contribution of one source at a point.
  - Sum contributions from all sources at a grid of points.
"""

# TODO Day 1: implement field computation