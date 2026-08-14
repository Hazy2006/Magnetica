from magnetica.field import Magnet, check_no_overlap
from magnetica.render import render_interactive


def main():
    magnets = [
        Magnet(position=(-1, 1), moment=(0, 1)),
        Magnet(position=(1, -1), moment=(3, 3)),
        Magnet(position=(1, 1), moment=(-2, 0)),
        Magnet(position=(-1, -1), moment=(1, -1))
    ]
    check_no_overlap(magnets)

    render_interactive(magnets, title="Magnetica — field of one dipole")


if __name__ == "__main__":
    main()