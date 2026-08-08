import numpy as np


class Magnet:
    def __init__(self, position, moment):
        self.position = np.array(position, dtype=float)
        self.moment = np.array(moment, dtype=float)


def compute_dipole_field(magnet, x, y):
    rx = x - magnet.position[0]
    ry = y - magnet.position[1]

    r_sq = rx ** 2 + ry ** 2
    # Safety mechanism to prevent divide-by-zero
    r_sq[r_sq == 0] = 1e-9
    r_mag = np.sqrt(r_sq)

    # Dot product of the magnetic moment and the distance vector
    dot_product = magnet.moment[0] * rx + magnet.moment[1] * ry

    # The magnetic dipole formulas
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