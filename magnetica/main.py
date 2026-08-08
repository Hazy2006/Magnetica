import numpy as np
import matplotlib.pyplot as plt
from magnetica.field import Magnet, compute_total_field


def main():

    xs = np.linspace(-5, 5, 25)
    ys = np.linspace(-5, 5, 25)
    x, y = np.meshgrid(xs, ys)

    magnets = [
        Magnet(position=(0, 0), moment=(0, 1)),
        Magnet(position=(1, -1), moment=(3, 3)),
        Magnet(position=(3, -1), moment=(-2, 0)),
    ]

    for magnet in magnets:
        strength = np.sqrt(magnet.moment[0] ** 2 + magnet.moment[1] ** 2)
        size = 50 + strength * 40  # base size 50, grows with strength
        plt.scatter(magnet.position[0], magnet.position[1],
                    c='red', s=size, edgecolors='black', zorder=5)

    bx, by = compute_total_field(magnets, x, y)
    magnitude = np.sqrt(bx ** 2 + by ** 2)
    bx_norm = bx / magnitude
    by_norm = by / magnitude



    plt.quiver(x, y, bx_norm, by_norm, magnitude, cmap='viridis')
    plt.colorbar(label='field strength')
    plt.gca().set_aspect('equal')   # so the plot isn't stretched
    plt.title("Magnetica — field of one dipole")
    plt.show()


if __name__ == "__main__":
    main()