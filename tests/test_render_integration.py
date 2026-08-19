import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace

import numpy as np

from magnetica.field import Magnet
from magnetica.render import (
    InteractiveScene, PARTICLE_STEPS_PER_FRAME, SPEED_BOOST_FACTOR,
    MODE_ATTRACTION, MODE_LORENTZ,
)


def _mouse_event(inaxes=None, button=None):
    return SimpleNamespace(inaxes=inaxes, xdata=None, ydata=None, button=button)


def _key_event(key):
    return SimpleNamespace(key=key)


def test_scene_has_three_axes_and_an_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    assert scene.ax_main is not None
    assert scene.ax_lap is not None
    assert scene.ax_energy is not None
    assert scene.analyzer is not None


def test_scene_ticks_without_error_and_analyzer_updates():
    # Two asymmetric magnets so the centroid (where the particle now
    # spawns at rest) isn't a force-equilibrium point -- a single magnet
    # at the origin would leave the particle motionless, since it would
    # spawn exactly on top of its own centroid with zero net offset.
    magnets = [Magnet(position=(-1, 0), moment=(0, 3), radius=0.1),
               Magnet(position=(1, 0), moment=(0, 5), radius=0.1)]
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


def test_pressing_speed_button_engages_boost_and_release_clears_it():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    assert scene.speed_boost is False
    scene.on_press(_mouse_event(inaxes=scene.ax_speed, button=1))
    assert scene.speed_boost is True

    scene.on_release(_mouse_event())
    assert scene.speed_boost is False


def test_pressing_elsewhere_does_not_engage_speed_boost():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    scene.on_press(_mouse_event(inaxes=scene.ax_main, button=1))
    assert scene.speed_boost is False


def test_speed_boost_runs_more_substeps_so_graphs_get_more_data():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]

    normal = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    normal._tick(0)
    normal_samples = len(normal.analyzer.energy_history["t"])

    boosted = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    boosted.speed_boost = True
    boosted._tick(0)
    boosted_samples = len(boosted.analyzer.energy_history["t"])

    assert normal_samples == PARTICLE_STEPS_PER_FRAME
    assert boosted_samples == round(PARTICLE_STEPS_PER_FRAME * SPEED_BOOST_FACTOR)
    assert boosted_samples > normal_samples


def test_scene_starts_in_attraction_mode():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    assert scene.mode == MODE_ATTRACTION


def test_pressing_t_toggles_mode_and_resets_trail_and_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(50):
        scene._tick(i)
    assert len(scene.trail_x) > 0
    assert len(scene.analyzer.energy_history["t"]) > 0

    scene.on_key(_key_event('t'))

    assert scene.mode == MODE_LORENTZ
    assert len(scene.trail_x) == 0
    assert len(scene.analyzer.energy_history["t"]) == 0

    scene.on_key(_key_event('t'))
    assert scene.mode == MODE_ATTRACTION


def test_lorentz_mode_ticks_without_error_and_speed_stays_bounded():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.on_key(_key_event('t'))
    assert scene.mode == MODE_LORENTZ

    initial_speed = np.linalg.norm(scene.particle.velocity)
    max_speed = initial_speed
    for i in range(100):
        scene._tick(i)
        max_speed = max(max_speed, np.linalg.norm(scene.particle.velocity))
        assert -5 <= scene.particle.position[0] <= 5
        assert -5 <= scene.particle.position[1] <= 5

    assert max_speed < initial_speed * 10


def test_toggling_mode_keeps_drag_and_scroll_handlers_working():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.on_key(_key_event('t'))
    assert scene.mode == MODE_LORENTZ

    scene.dragged_index = 0
    scene.on_motion(SimpleNamespace(inaxes=scene.ax_main, xdata=1.5, ydata=1.5, button=1))
    assert np.allclose(scene.magnets[0].position, [1.5, 1.5])

    strength_before = np.linalg.norm(scene.magnets[0].moment)
    scene.on_scroll(SimpleNamespace(inaxes=scene.ax_main, xdata=1.5, ydata=1.5, button='up'))
    assert np.linalg.norm(scene.magnets[0].moment) > strength_before


def test_adding_a_magnet_rebaselines_the_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(100):
        scene._tick(i)
    assert len(scene.analyzer.energy_history["t"]) > 0

    scene._try_add_magnet(2.0, 2.0)

    assert len(scene.magnets) == 2
    assert scene.analyzer.energy_history["t"] == []
    assert scene.analyzer.energy_history["rel_err"] == []


def test_removing_a_magnet_rebaselines_the_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1),
               Magnet(position=(2, 2), moment=(0, 1), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(100):
        scene._tick(i)
    assert len(scene.analyzer.energy_history["t"]) > 0

    scene.on_press(SimpleNamespace(inaxes=scene.ax_main, xdata=2.0, ydata=2.0, button=3))

    assert len(scene.magnets) == 1
    assert scene.analyzer.energy_history["t"] == []


def test_lorentz_mode_stops_and_flags_escape_when_particle_leaves_the_frame():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.on_key(_key_event('t'))
    assert scene.mode == MODE_LORENTZ

    # Aim it straight at the wall, fast enough to guarantee it exits on tick 1.
    scene.particle.position = np.array([4.99, 0.0])
    scene.particle.velocity = np.array([50.0, 0.0])

    scene._tick(0)

    assert scene.escaped is True
    assert scene.particle.position[0] == 5
    assert "ESCAPED" in scene.escape_text.get_text()

    frozen_position = scene.particle.position.copy()
    history_len = len(scene.analyzer.energy_history["t"])
    scene._tick(1)

    assert np.array_equal(scene.particle.position, frozen_position)
    assert len(scene.analyzer.energy_history["t"]) == history_len


def test_toggling_mode_recovers_from_an_escape():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.on_key(_key_event('t'))
    scene.particle.position = np.array([4.99, 0.0])
    scene.particle.velocity = np.array([50.0, 0.0])
    scene._tick(0)
    assert scene.escaped is True

    scene.on_key(_key_event('t'))
    assert scene.escaped is False
    assert scene.mode == MODE_ATTRACTION

    scene._tick(1)
    assert scene.escape_text.get_text() == ''


def test_attraction_mode_wall_contact_bounces_instead_of_escaping():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    assert scene.mode == MODE_ATTRACTION

    scene.particle.position = np.array([4.99, 0.0])
    scene.particle.velocity = np.array([50.0, 0.0])

    scene._tick(0)

    assert scene.escaped is False
    assert scene.escape_text.get_text() == ''
    assert -5 <= scene.particle.position[0] <= 5
