import matplotlib
matplotlib.use("Agg")

import numpy as np

from magnetica.field import Magnet
from magnetica.render import InteractiveScene


def test_scene_has_three_axes_and_an_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    assert scene.ax_main is not None
    assert scene.ax_poincare is not None
    assert scene.ax_energy is not None
    assert scene.analyzer is not None


def test_scene_ticks_without_error_and_analyzer_updates():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(200):
        scene._tick(i)

    assert scene.analyzer.status in ("Bounded", "Unbound")
    assert scene.analyzer.max_radius > 0
    hist = scene.analyzer.energy_history
    assert len(hist["t"]) > 0


def test_scene_particle_still_respects_wall_bounds_with_analyzer_running():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.particle.position = np.array([0.5, 2.0])
    scene.particle.velocity = np.array([0.0, -1.0])

    for i in range(400):
        scene._tick(i)
        assert -5 <= scene.particle.position[0] <= 5
        assert -5 <= scene.particle.position[1] <= 5


def test_analyzer_table_and_side_plots_update_after_ticks():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(600):
        scene._tick(i)

    assert scene.analyzer_text.get_text() != ""
    assert "Status" in scene.analyzer_text.get_text()
    assert len(scene.ke_line.get_xdata()) > 0
    assert len(scene.rel_err_line.get_xdata()) == len(scene.ke_line.get_xdata())
