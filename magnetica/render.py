"""
Visualization — turning field + particle into something you can see.

Day 1: static field render (arrows / coloured dots).
Day 2: colour by strength, multiple sources.
Day 4: animate the particle in real time.
Day 5: drag magnets around and watch the field update live.
Day 6: add/remove/rescale magnets and a continuously animated particle,
       all driven from one interactive scene.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import PowerNorm
from magnetica.field import Magnet, compute_total_field, check_no_overlap, OverlappingMagnetsError
from magnetica.particle import Particle
from magnetica.analysis import OrbitAnalyzer

PICK_RADIUS = 0.4       # data-space distance within which a click/scroll grabs a magnet
DEFAULT_MOMENT = (0.0, 1.0)
SCROLL_FACTOR = 1.15     # multiplicative strength change per scroll notch
MIN_STRENGTH = 0.1       # floor so a magnet's direction never becomes undefined

PARTICLE_CHARGE = 1.0
PARTICLE_DT = 0.01
PARTICLE_STEPS_PER_FRAME = 5
TRAIL_LENGTH = 300
ATTRACTION_STRENGTH = 0.5   # g in Particle.step_attracted; tuned so orbits stay
                            # bounded well within the default (-5, 5) view
PARTICLE_ORBIT_OFFSET = np.array([2.5, 0.0])  # start this far from the magnets' centroid


def default_particle_start(magnets, g=ATTRACTION_STRENGTH, offset=PARTICLE_ORBIT_OFFSET):
    """A starting position/velocity for a roughly circular orbit around
    the magnets' centroid, treating their combined strength as a single
    point mass at that starting radius. Multiple magnets and an
    off-center start mean this is only an approximation, not an exact
    solution -- but it's close enough to launch the particle into a
    bounded orbit instead of straight into a magnet or off to infinity."""
    centroid = magnet_positions(magnets).mean(axis=0) if magnets else np.zeros(2)
    total_strength = sum(np.linalg.norm(m.moment) for m in magnets) or 1.0
    radius = np.linalg.norm(offset)
    speed = np.sqrt(g * total_strength / radius)
    tangent = np.array([-offset[1], offset[0]]) / radius
    return centroid + offset, tangent * speed

INSTRUCTIONS = (
    "Left-drag: move magnet    |    a, then click: add magnet    |    "
    "Right-click: remove magnet    |    Scroll over a magnet: adjust strength"
)


def magnet_marker_size(magnet):
    strength = np.linalg.norm(magnet.moment)
    return 50 + strength * 40  # base size 50, grows with strength


def magnet_positions(magnets):
    if not magnets:
        return np.empty((0, 2))
    return np.array([m.position for m in magnets])


def compute_field_arrows(magnets, x, y):
    bx, by = compute_total_field(magnets, x, y)
    magnitude = np.sqrt(bx ** 2 + by ** 2)

    # Safely normalize to prevent divide by zero where magnitude is 0
    mag_safe = np.where(magnitude == 0, 1e-9, magnitude)
    bx_norm = bx / mag_safe
    by_norm = by / mag_safe
    return bx_norm, by_norm, magnitude


def strength_norm(magnitude):
    # Field strength falls off as 1/r^3, so a linear scale makes almost
    # everything read as the same low colour except right at a magnet.
    # PowerNorm compresses that range so the falloff halo stays visible,
    # and capping vmax at a percentile keeps the singularity at a magnet's
    # center from blowing out the rest of the scale.
    vmax = max(np.percentile(magnitude, 99), 1e-9)
    return PowerNorm(gamma=0.35, vmin=0, vmax=vmax)


class InteractiveScene:
    """Owns the whole interactive plot: the magnets, the field display,
    the animated particle, and every mouse/keyboard interaction. A single
    timer tick redraws everything from current state, so handlers only
    need to mutate state (move/add/remove/rescale a magnet) rather than
    each managing their own redraw."""

    def __init__(self, magnets, xlim, ylim, resolution, title):
        check_no_overlap(magnets)
        self.magnets = magnets
        self.xlim = xlim
        self.ylim = ylim
        xs = np.linspace(*xlim, resolution)
        ys = np.linspace(*ylim, resolution)
        self.x, self.y = np.meshgrid(xs, ys)

        self.dragged_index = None
        self.add_armed = False

        start_position, start_velocity = default_particle_start(magnets)
        self.particle = Particle(position=start_position,
                                  velocity=start_velocity,
                                  charge=PARTICLE_CHARGE)
        self.trail_x = []
        self.trail_y = []
        self.sim_time = 0.0
        self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT)

        self._build_figure(title)
        self._connect_events()

        self.animation = animation.FuncAnimation(
            self.fig, self._tick, interval=20, blit=False, cache_frame_data=False)

    def _build_figure(self, title):
        bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y)

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1],
            left=0.06, right=0.97, top=0.93, bottom=0.18,
            wspace=0.30, hspace=0.40,
        )
        self.ax_poincare = self.fig.add_subplot(gs[0, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_main = self.fig.add_subplot(gs[:, 1])

        self.mesh = self.ax_main.pcolormesh(self.x, self.y, magnitude, cmap='inferno',
                                             shading='gouraud', norm=strength_norm(magnitude), zorder=0)
        self.quiver = self.ax_main.quiver(self.x, self.y, bx_norm, by_norm,
                                           color='white', alpha=0.8, zorder=2)
        self.scatter = self.ax_main.scatter(*magnet_positions(self.magnets).T,
                                             c='red', s=[magnet_marker_size(m) for m in self.magnets],
                                             edgecolors='black', zorder=5, picker=True)

        self.trail_line, = self.ax_main.plot([], [], color='deepskyblue', alpha=0.6, zorder=4)
        self.particle_dot, = self.ax_main.plot([], [], 'go', markersize=8, zorder=6)

        self.fig.colorbar(self.mesh, ax=self.ax_main, label='field strength')
        self.ax_main.set_xlim(*self.xlim)
        self.ax_main.set_ylim(*self.ylim)
        self.ax_main.set_xlabel('x')
        self.ax_main.set_ylabel('y')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title(title)

        self.ax_main.legend([self.scatter, self.particle_dot, self.trail_line],
                             ['magnet', 'particle', 'trail'],
                             loc='upper right', fontsize=8, framealpha=0.7)

        self.status_text = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes, va='top', ha='left',
            fontsize=9, color='yellow', bbox=dict(facecolor='black', alpha=0.5, pad=3))

        self.fig.text(0.5, 0.02, INSTRUCTIONS, ha='center', va='bottom', fontsize=9)

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    # --- picking helpers ---

    def _nearest_magnet(self, xdata, ydata):
        if not self.magnets:
            return None
        distances = [np.hypot(m.position[0] - xdata, m.position[1] - ydata)
                     for m in self.magnets]
        index = int(np.argmin(distances))
        if distances[index] <= PICK_RADIUS:
            return index
        return None

    def _resolve_overlap(self, index, candidate):
        """Push `candidate` back outside every other magnet's radius, so a
        drag can slide a magnet up to its neighbours but never through
        or on top of them."""
        magnet = self.magnets[index]
        for j, other in enumerate(self.magnets):
            if j == index:
                continue
            min_dist = magnet.radius + other.radius
            offset = candidate - other.position
            dist = np.linalg.norm(offset)
            if dist < min_dist:
                if dist == 0:
                    offset = magnet.position - other.position
                    dist = np.linalg.norm(offset)
                    if dist == 0:
                        offset = np.array([1.0, 0.0])
                        dist = 1.0
                candidate = other.position + offset / dist * min_dist
        return candidate

    # --- event handlers: mutate state only, the animation tick redraws ---

    def on_press(self, event):
        if event.inaxes != self.ax_main or event.xdata is None:
            return

        if event.button == 3:  # right-click: remove
            index = self._nearest_magnet(event.xdata, event.ydata)
            if index is not None:
                if self.dragged_index == index:
                    self.dragged_index = None
                del self.magnets[index]
            return

        if event.button == 1:
            if self.add_armed:
                self._try_add_magnet(event.xdata, event.ydata)
                self.add_armed = False
                return
            self.dragged_index = self._nearest_magnet(event.xdata, event.ydata)

    def _try_add_magnet(self, xdata, ydata):
        candidate = Magnet(position=(xdata, ydata), moment=DEFAULT_MOMENT)
        try:
            check_no_overlap(self.magnets + [candidate])
        except OverlappingMagnetsError:
            return  # spot is taken; ignore the click rather than crash
        self.magnets.append(candidate)

    def on_motion(self, event):
        if self.dragged_index is None or event.inaxes != self.ax_main or event.xdata is None:
            return
        candidate = np.array([event.xdata, event.ydata])
        candidate = self._resolve_overlap(self.dragged_index, candidate)
        self.magnets[self.dragged_index].position = candidate

    def on_release(self, event):
        self.dragged_index = None

    def on_scroll(self, event):
        if event.inaxes != self.ax_main or event.xdata is None:
            return
        index = self._nearest_magnet(event.xdata, event.ydata)
        if index is None:
            return
        magnet = self.magnets[index]
        strength = np.linalg.norm(magnet.moment)
        direction = magnet.moment / strength if strength > 0 else np.array([0.0, 1.0])
        factor = SCROLL_FACTOR if event.button == 'up' else 1 / SCROLL_FACTOR
        new_strength = max(strength * factor, MIN_STRENGTH)
        magnet.moment = direction * new_strength

    def on_key(self, event):
        if event.key == 'a':
            self.add_armed = not self.add_armed

    # --- redraw ---

    def _status_message(self):
        if self.add_armed:
            return "Add mode armed — click anywhere to place a magnet"
        return ''

    def _tick(self, frame):
        self.status_text.set_text(self._status_message())

        positions = magnet_positions(self.magnets)
        self.scatter.set_offsets(positions)
        self.scatter.set_sizes([magnet_marker_size(m) for m in self.magnets])

        bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y)
        self.mesh.set_array(magnitude.ravel())
        self.mesh.set_norm(strength_norm(magnitude))
        self.quiver.set_UVC(bx_norm, by_norm)

        for _ in range(PARTICLE_STEPS_PER_FRAME):
            self.particle.step_attracted(self.magnets, PARTICLE_DT, g=ATTRACTION_STRENGTH)
            self.particle.clamp_to_bounds(self.xlim, self.ylim)
            self.sim_time += PARTICLE_DT
            self.analyzer.update(self.sim_time, self.particle.position, self.particle.velocity, self.magnets)

        self.trail_x.append(self.particle.position[0])
        self.trail_y.append(self.particle.position[1])
        if len(self.trail_x) > TRAIL_LENGTH:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.particle_dot.set_data([self.particle.position[0]], [self.particle.position[1]])

        return (self.mesh, self.quiver, self.scatter,
                self.trail_line, self.particle_dot, self.status_text)

    def show(self):
        plt.show()


def render_interactive(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=25, title="Magnetica — field of magnets"):
    """Render magnets and their combined field, with a continuously
    animated particle and full mouse/keyboard control: drag magnets to
    move them, press 'a' then click to add one, right-click to remove
    one, and scroll over a magnet to adjust its strength."""
    scene = InteractiveScene(magnets, xlim, ylim, resolution, title)
    scene.show()
    return scene
