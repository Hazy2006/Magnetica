from magnetica.field import Magnet, check_no_overlap
from magnetica.render import render_interactive


def main():
    magnets = [
        Magnet(position=(-2.0, 2.0), moment=(1, 1)),
        Magnet(position=(2.0, 2.0), moment=(-1.2, 1)),
        Magnet(position=(2.5, -1.9), moment=(1, 1.3)),
        Magnet(position=(-2.0, -2.0), moment=(-1, -1)),
    ]
    check_no_overlap(magnets)

    render_interactive(magnets)


if __name__ == "__main__":
    main()