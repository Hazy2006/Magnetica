import numpy as np


class OverlappingMagnetsError(ValueError):
    """Raised when two magnets' physical extents (position + radius) overlap."""


def check_no_overlap(magnets):
    for i in range(len(magnets)):
        for j in range(i + 1, len(magnets)):
            dist = np.linalg.norm(magnets[i].position - magnets[j].position)
            min_dist = magnets[i].radius + magnets[j].radius
            if dist < min_dist:
                raise OverlappingMagnetsError(
                    "magnets overlap, please change coordinates in the code"
                )


class Magnet:
    def __init__(self, position, moment, radius=0.1):
        self.position = np.array(position, dtype=float)
        self.moment = np.array(moment, dtype=float)
        self.radius = radius

def compute_dipole_field(magnet, x, y):
    rx = x - magnet.position[0]
    ry = y - magnet.position[1]

    r_sq = rx ** 2 + ry ** 2
    r_sq[r_sq == 0] = 1e-9  # avoid divide-by-zero at the magnet's center
    r_mag = np.sqrt(r_sq)

    dot_product = magnet.moment[0] * rx + magnet.moment[1] * ry

    # 2D magnetic dipole field
    bx = (3 * rx * dot_product / r_mag ** 5) - (magnet.moment[0] / r_mag ** 3)
    by = (3 * ry * dot_product / r_mag ** 5) - (magnet.moment[1] / r_mag ** 3)

    return bx, by

def compute_total_field(magnets, x, y):
    bx_total = np.zeros_like(x)
    by_total = np.zeros_like(y)

    for magnet in magnets:
        bx, by = compute_dipole_field(magnet, x, y)
        bx_total += bx
        by_total += by

    return bx_total, by_total

def compute_bz(magnets, x, y):
    """
    Out-of-plane field strength (z-component) at each point.
    Each magnet contributes strength / (r^2 + radius^2), falling off
    with distance while accounting for the physical size of the magnet.
    """
    bz_total = np.zeros_like(x, dtype=float)
    for magnet in magnets:
        rx = x - magnet.position[0]
        ry = y - magnet.position[1]
        r_sq = rx**2 + ry**2
        strength = np.sqrt(magnet.moment[0]**2 + magnet.moment[1]**2)
        bz_total += strength / (r_sq + magnet.radius**2)
    return bz_total

def compute_attraction_field(magnets, x, y, g):
    """Vectorized form of Particle._attraction_acceleration's central-pull
    formula, over a grid -- drives the attraction-mode field display so it
    matches the force actually steering the particle, instead of the
    unrelated dipole field."""
    fx_total = np.zeros_like(x, dtype=float)
    fy_total = np.zeros_like(y, dtype=float)
    for magnet in magnets:
        offset_x = magnet.position[0] - x
        offset_y = magnet.position[1] - y
        strength = np.sqrt(magnet.moment[0] ** 2 + magnet.moment[1] ** 2)
        r_sq = offset_x ** 2 + offset_y ** 2
        softened = (r_sq + magnet.radius ** 2) ** 1.5
        fx_total += g * strength * offset_x / softened
        fy_total += g * strength * offset_y / softened
    return fx_total, fy_total